//! Session connection budgets matching aiohttp's connector limits.
use std::{sync::Arc, time::Duration};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
#[derive(Clone)]
pub struct Limit {
    slots: Arc<Semaphore>,
    timeout: Duration,
}
impl Limit {
    pub fn new(count: usize, seconds: f64) -> Self {
        Self {
            slots: Arc::new(Semaphore::new(count)),
            timeout: Duration::from_secs_f64(seconds),
        }
    }
    pub async fn acquire(&self) -> anyhow::Result<OwnedSemaphorePermit> {
        Ok(tokio::time::timeout(self.timeout, self.slots.clone().acquire_owned()).await??)
    }
}
