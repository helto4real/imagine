pub mod backend;
pub mod comfy;
pub mod config;
pub mod coordinator;
pub mod http;
pub mod logging;
pub mod profiles;
pub mod requests;
pub mod routes;
pub mod service;

pub mod workflows;

pub mod privacy;

pub mod imagine;

pub mod references;

pub mod mcp;

pub mod diagnostics;

pub mod request_log;

pub mod json_wire;

pub mod mcp_protocol;

pub mod connection_limit;

mod mcp_storage;
#[path = "../../shared/workspace_ids.rs"]
pub mod workspace_ids;

mod coordinator_socket;

#[path = "../../shared/video_destination.rs"]
pub mod media_destination;

#[cfg(test)]
#[path = "../../shared/task_fixtures.rs"]
mod task_fixtures;
