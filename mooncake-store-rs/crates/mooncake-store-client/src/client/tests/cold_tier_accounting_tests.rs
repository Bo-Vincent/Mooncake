use super::*;

fn materialize_accounting_routes(
    client: &StoreClient,
    routes: &[ObjectRoute],
) -> mooncake_store_core::Result<usize> {
    let entries = routes
        .iter()
        .map(|route| {
            client
                .storage_owner
                .pending_offloads
                .claim(&route.key, route.version)
                .expect("test should claim its pending route")
        })
        .collect();
    let pending = client
        .storage_owner
        .prepare_pending_offload_entries(entries)?;
    client
        .storage_owner
        .materialize_prepared_pending_offload_batch(&pending, &mut None)
}

#[test]
fn cold_offload_rejected_publication_releases_usage() {
    let _cold_tier_env = enable_cold_tier_for_test();
    for reject_charge in [false, true] {
        let rejected_key = ObjectKey::from_scope(
            &NamespaceScope::with_defaults(Some("default"), None, None),
            "rejected-key",
        );
        let metadata = Arc::new(BlockingCasMetadataBackend::new(
            Arc::new(InMemoryMetadataBackend::new()),
            &rejected_key.0,
        ));
        metadata
            .reject_cold_charge
            .store(reject_charge, Ordering::SeqCst);
        metadata
            .reject_materialized_publication
            .store(!reject_charge, Ordering::SeqCst);
        let device_id = if reject_charge {
            "cold-charge-failure"
        } else {
            "cold-publication-conflict"
        };
        let root = cold_tier_test_root(device_id);
        let mut client = StoreClientBuilder::new(metadata.clone(), device_id)
            .route_control(RouteControlMode::MetadataOnly)
            .state(ClientLifecycleState::Active)
            .label("storage", "true")
            .transport(Arc::new(TestTransport::new(device_id)))
            .local_memory(storage_config_with_bytes(512))
            .cold_tier_target(ColdTierTargetConfig::directory(
                device_id,
                ColdTierKind::Ssd,
                root.clone(),
            ))
            .build(test_future_expiry_ms())
            .expect("client should build");
        client
            .register_local_memory()
            .expect("memory should register");
        client.cold_tier.shutdown();
        let payload = b"cold-rejected-payload";
        client
            .put("rejected-key", payload)
            .expect("put should succeed");
        let route = client.query_route("rejected-key").unwrap().unwrap();
        let backing = route
            .cold_backing
            .as_ref()
            .expect("pending backing should exist");
        let mut routes = vec![route.clone()];
        let accepted_bytes = if reject_charge {
            0
        } else {
            client
                .put("accepted-key", b"accepted")
                .expect("put should succeed");
            routes.push(client.query_route("accepted-key").unwrap().unwrap());
            8
        };
        let result = materialize_accounting_routes(&client, &routes);
        if reject_charge {
            assert!(result.is_err(), "failed charge must prevent publication");
        } else {
            assert_eq!(result.expect("conflict should be handled"), 1);
        }
        let device = metadata.get_cold_tier_device(device_id).unwrap().unwrap();
        assert_eq!(
            device.used_bytes, accepted_bytes,
            "only successful publications should retain used bytes"
        );
        assert_eq!(
            device.reserved_bytes, 0,
            "failed publication must release reservations"
        );
        assert!(
            client
                .storage_owner
                .cold_tier_devices
                .backend_for(backing)
                .unwrap()
                .get_object(backing)
                .unwrap()
                .is_none(),
            "unpublished payload must be removed"
        );
        let current = client.query_route("rejected-key").unwrap().unwrap();
        assert_eq!(
            current.cold_backing.unwrap().state,
            mooncake_store_core::ColdBackingState::PendingOffload
        );
        assert_eq!(client.get("rejected-key").unwrap(), payload);
        if reject_charge {
            metadata.reject_cold_charge.store(false, Ordering::SeqCst);
            assert_eq!(
                materialize_accounting_routes(&client, std::slice::from_ref(&route))
                    .expect("retry should materialize"),
                1
            );
            let device = metadata.get_cold_tier_device(device_id).unwrap().unwrap();
            assert_eq!(device.used_bytes, payload.len() as u64);
            assert_eq!(device.reserved_bytes, 0);
        }
        drop(client);
        let _ = std::fs::remove_dir_all(root);
    }
}

struct ReleasePublication<'a>(&'a BlockingCasMetadataBackend);

impl Drop for ReleasePublication<'_> {
    fn drop(&mut self) {
        self.0.release_blocked_cas();
    }
}

fn run_cold_accounting_case(delete_before_settlement: bool) -> (u64, u64) {
    let _cold_tier_env = enable_cold_tier_for_test();
    let metadata = Arc::new(BlockingCasMetadataBackend::new(
        Arc::new(InMemoryMetadataBackend::new()),
        "unused-cas-gate",
    ));
    metadata.arm_blocked_cas();
    metadata
        .block_materialized_publication
        .store(true, Ordering::SeqCst);
    let device_id = if delete_before_settlement {
        "community-cold-race"
    } else {
        "community-cold-control"
    };
    let root = cold_tier_test_root(device_id);
    let client = StoreClientBuilder::new(metadata.clone(), device_id)
        .route_control(RouteControlMode::MetadataOnly)
        .state(ClientLifecycleState::Active)
        .label("storage", "true")
        .transport(Arc::new(TestTransport::new(device_id)))
        .local_memory(storage_config_with_bytes(512))
        .cold_tier_target(ColdTierTargetConfig::directory(
            device_id,
            ColdTierKind::Ssd,
            root.clone(),
        ))
        .build(test_future_expiry_ms())
        .expect("client should build");
    client
        .register_local_memory()
        .expect("memory should register");
    let release = ReleasePublication(&metadata);
    client
        .put("settlement-race-key", b"cold-delete-payload")
        .expect("public put should succeed");
    assert!(
        metadata.wait_until_blocked(Duration::from_secs(5)),
        "real Materialized publication must reach the gate"
    );
    let backing = wait_for_materialized_cold_backing(&client, "settlement-race-key");
    let visible_used = metadata
        .get_cold_tier_device(device_id)
        .unwrap()
        .unwrap()
        .used_bytes;
    assert_eq!(backing.length, 19);
    if delete_before_settlement {
        client
            .remove("settlement-race-key", true)
            .expect("public remove should succeed");
        let _ = client.flush_due_reclaims();
    }
    drop(release);
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if !client.storage_owner.pending_offloads.is_materializing()
            && client.storage_owner.pending_offloads.is_empty()
        {
            break;
        }
        assert!(Instant::now() < deadline, "offload settlement must finish");
        std::thread::yield_now();
    }
    if !delete_before_settlement {
        let settled = metadata.get_cold_tier_device(device_id).unwrap().unwrap();
        assert_eq!(
            settled.used_bytes, 19,
            "control must finish charging before delete"
        );
        client
            .remove("settlement-race-key", true)
            .expect("public remove should succeed");
    }
    let reclaim_result = client.flush_due_reclaims();
    let route = client.query_route("settlement-race-key").unwrap();
    let payload_exists = client
        .storage_owner
        .cold_tier_devices
        .backend_for(&backing)
        .unwrap()
        .get_object(&backing)
        .unwrap()
        .is_some();
    let final_device = metadata.get_cold_tier_device(device_id).unwrap().unwrap();
    eprintln!("delete_before_settlement={delete_before_settlement} visible_used={visible_used} final_used={} reserved={} route={route:?} payload_exists={payload_exists} reclaim={reclaim_result:?}", final_device.used_bytes, final_device.reserved_bytes);
    assert!(
        route.is_none(),
        "public remove must delete the authoritative route"
    );
    assert!(
        !payload_exists,
        "public reclaim must delete the cold payload"
    );
    assert_eq!(final_device.reserved_bytes, 0);
    let final_used = final_device.used_bytes;
    drop(client);
    let _ = std::fs::remove_dir_all(root);
    (visible_used, final_used)
}

#[test]
fn materialized_route_has_accounted_usage() {
    let (visible_used, _) = run_cold_accounting_case(false);
    assert!(
        visible_used >= 19,
        "Materialized route must have charged payload: used={visible_used} length=19"
    );
}

#[test]
fn delete_during_publication_reclaims_usage() {
    let (_, final_used) = run_cold_accounting_case(true);
    assert_eq!(
        final_used, 0,
        "deleted cold payload must not leave charged bytes"
    );
}

#[test]
fn delete_after_settlement_reclaims_usage() {
    let (_, final_used) = run_cold_accounting_case(false);
    assert_eq!(
        final_used, 0,
        "settled delete control must release charged bytes"
    );
}
