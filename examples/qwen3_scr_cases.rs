#[path = "support/qwen3_cases.rs"]
mod qwen3_cases;

use std::env;

use heterogeneous_weight_conversion::ManifestWeightConversionPlugin;

fn usage() -> ! {
    eprintln!(
        "usage:\n  qwen3_scr_cases --list\n  \
         qwen3_scr_cases <case-name> [request|plan]"
    );
    std::process::exit(2);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.as_slice() == ["--list"] {
        for case in qwen3_cases::all_cases() {
            println!("{}\t{}", case.name, case.description);
        }
        return Ok(());
    }
    if args.is_empty() || args.len() > 2 {
        usage();
    }

    let case = qwen3_cases::find_case(&args[0]).unwrap_or_else(|| {
        eprintln!("unknown Qwen3 case: {}", args[0]);
        usage();
    });
    let output = args.get(1).map(String::as_str).unwrap_or("request");
    let request = case.request();

    match output {
        "request" => println!("{}", request.to_json()?),
        "plan" => println!(
            "{}",
            ManifestWeightConversionPlugin::default()
                .plan_scr(&request, None)?
                .to_json()?
        ),
        _ => usage(),
    }
    Ok(())
}
