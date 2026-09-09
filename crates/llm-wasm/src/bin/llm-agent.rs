//! Native CLI for llm-wasm. `required-features = ["native"]` in Cargo.toml
//! keeps this out of the wasm32-unknown-unknown / wasm-pack build.

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "llm-agent", about = "xLAM-2-3b-fc-r native CLI (skeleton)")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run the agent loop against a prompt.
    Run,
    /// Evaluate the model against reference logits/outputs.
    Eval,
    /// Print GGUF header/tensor info for a model file.
    GgufInfo,
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run => println!("run: not implemented"),
        Commands::Eval => println!("eval: not implemented"),
        Commands::GgufInfo => println!("gguf-info: not implemented"),
    }
}
