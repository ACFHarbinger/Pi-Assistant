use crate::tools::{PermissionTier, Tool, ToolContext, ToolResult};
use anyhow::Result;
use async_trait::async_trait;
use serde_json::{json, Value};

pub struct KnowledgeGraphTool;

impl KnowledgeGraphTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for KnowledgeGraphTool {
    fn name(&self) -> &str {
        "knowledge_graph"
    }

    fn description(&self) -> &str {
        "Interact with the structured knowledge graph to store and retrieve entity relationships.
Actions:
- upsert_entity: Create or update an entity (returns UUID).
- add_relation: Create a relationship between two entity UUIDs.
- find_related: Find entities related to a given name."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["upsert_entity", "add_relation", "find_related"],
                    "description": "The action to perform"
                },
                "name": {
                    "type": "string",
                    "description": "Entity name (for upsert/find)"
                },
                "entity_type": {
                    "type": "string",
                    "description": "Entity type (e.g., 'file', 'person', 'concept')"
                },
                "metadata": {
                    "type": "object",
                    "description": "JSON metadata for the entity"
                },
                "from_id": {
                    "type": "string",
                    "description": "Source entity UUID (for relation)"
                },
                "to_id": {
                    "type": "string",
                    "description": "Target entity UUID (for relation)"
                },
                "relation_type": {
                    "type": "string",
                    "description": "Type of relation (e.g., 'depends_on')"
                },
                "weight": {
                    "type": "number",
                    "description": "Relation weight (default 1.0)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results for find_related (default 10)"
                }
            },
            "required": ["action"]
        })
    }

    fn permission_tier(&self) -> PermissionTier {
        PermissionTier::Low // Generally safe to read/write knowledge
    }

    async fn execute(&self, params: Value, context: ToolContext) -> Result<ToolResult> {
        let action = params
            .get("action")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing action parameter"))?;

        let memory = context
            .memory
            .ok_or_else(|| anyhow::anyhow!("Memory manager not available in tool context"))?;

        match action {
            "upsert_entity" => {
                let name = params
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing 'name'"))?;
                let entity_type = params
                    .get("entity_type")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing 'entity_type'"))?;
                let metadata = params.get("metadata").cloned();

                let id = memory.upsert_entity(name, entity_type, metadata)?;
                Ok(ToolResult::success(
                    json!({ "id": id.to_string() }).to_string(),
                ))
            }
            "add_relation" => {
                let from_id = params
                    .get("from_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing 'from_id'"))?;
                let to_id = params
                    .get("to_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing 'to_id'"))?;
                let relation_type = params
                    .get("relation_type")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing 'relation_type'"))?;
                let weight = params.get("weight").and_then(|v| v.as_f64()).unwrap_or(1.0);

                let uuid_from = uuid::Uuid::parse_str(from_id)?;
                let uuid_to = uuid::Uuid::parse_str(to_id)?;

                let id = memory.add_relation(&uuid_from, &uuid_to, relation_type, weight)?;
                Ok(ToolResult::success(
                    json!({ "id": id.to_string() }).to_string(),
                ))
            }
            "find_related" => {
                let name = params
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing 'name'"))?;
                let limit = params.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

                let nodes = memory.find_related_entities(name, limit)?;
                Ok(ToolResult::success(serde_json::to_string(&nodes)?))
            }
            _ => Err(anyhow::anyhow!("Unknown action: {}", action)),
        }
    }
}
