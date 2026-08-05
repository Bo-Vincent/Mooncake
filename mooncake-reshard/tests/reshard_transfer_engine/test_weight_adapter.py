from __future__ import annotations

from mooncake.reshard.transfer_engine import MooncakeTransferEngineExecutor
from mooncake.reshard.weight import (
    MooncakeTransferEngineReader,
    MooncakeTransferEngineSink,
)


class FakeEngine:
    def get_engine_ptr(self) -> int:
        return id(self)


def test_weight_reader_and_sink_share_resource_neutral_executor() -> None:
    reader = MooncakeTransferEngineReader(FakeEngine())
    sink = MooncakeTransferEngineSink(FakeEngine())

    assert isinstance(reader.transfer_executor, MooncakeTransferEngineExecutor)
    assert isinstance(sink.transfer_executor, MooncakeTransferEngineExecutor)
