use std::env;
use std::io::{self, Read};
use std::process::ExitCode;

use heterogeneous_weight_conversion::ManifestWeightConversionPlugin;

fn parse_max_chunk_bytes() -> Result<Option<usize>, String> {
    let mut arguments = env::args().skip(1);
    let mut max_chunk_bytes = None;
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--max-chunk-bytes" => {
                let value = arguments
                    .next()
                    .ok_or_else(|| "--max-chunk-bytes requires a value".to_owned())?;
                let value = value
                    .parse::<usize>()
                    .map_err(|_| "--max-chunk-bytes must be a positive integer".to_owned())?;
                if value == 0 {
                    return Err("--max-chunk-bytes must be a positive integer".to_owned());
                }
                max_chunk_bytes = Some(value);
            }
            "--help" | "-h" => {
                println!(
                    "Usage: hweight-plan [--max-chunk-bytes N]\n\
                     Reads a ConversionRequest JSON object from stdin and writes \
                     a ScrTransferPlan JSON object to stdout."
                );
                std::process::exit(0);
            }
            _ => return Err(format!("unsupported argument: {argument}")),
        }
    }
    Ok(max_chunk_bytes)
}

fn run() -> Result<(), String> {
    let max_chunk_bytes = parse_max_chunk_bytes()?;
    let mut request = String::new();
    io::stdin()
        .read_to_string(&mut request)
        .map_err(|error| format!("failed to read stdin: {error}"))?;
    let plan = ManifestWeightConversionPlugin::default()
        .plan_scr_json(&request, max_chunk_bytes)
        .map_err(|error| error.to_string())?;
    println!("{plan}");
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("hweight-plan: {error}");
            ExitCode::FAILURE
        }
    }
}
