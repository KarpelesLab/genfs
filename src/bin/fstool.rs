//! fstool CLI — thin wrapper over the library.
//!
//! Subcommands will be wired up in P5 (clap). For now this is a placeholder
//! so `cargo build` produces a runnable binary while the library lands.

fn main() {
    eprintln!(
        "fstool {} — CLI not yet implemented (see roadmap)",
        env!("CARGO_PKG_VERSION")
    );
    std::process::exit(2);
}
