//! Dedicated one-job H3 product worker. Never used for the frozen replay.
fn main() {
    if !imagine_inference::h3_product::wire::quiet_entry(imagine_inference::h3_product::run) {
        std::process::exit(1);
    }
}
