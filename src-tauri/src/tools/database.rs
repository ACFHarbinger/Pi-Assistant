//! Database tool for direct SQL access to user-specified SQLite databases.

use super::{PermissionTier, Tool, ToolContext, ToolResult};
use anyhow::Result;
use async_trait::async_trait;
use rusqlite::{Connection, OpenFlags};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::info;

struct DbSession {
    conn: Connection,
    readonly: bool,
}

/// Database tool for SQL access to user-specified databases.
pub struct DatabaseTool {
    /// Active connections keyed by alias.
    connections: Arc<Mutex<HashMap<String, DbSession>>>,
}

impl DatabaseTool {
    pub fn new() -> Self {
        Self {
            connections: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Derive a default alias from a file path.
    fn default_alias(path: &str) -> String {
        PathBuf::from(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("db")
            .to_string()
    }

    async fn connect(&self, path: &str, alias: Option<&str>, readonly: bool) -> Result<ToolResult> {
        let alias = alias
            .map(|s| s.to_string())
            .unwrap_or_else(|| Self::default_alias(path));

        info!(path = %path, alias = %alias, readonly = %readonly, "Connecting to database");

        let path_buf = PathBuf::from(path);
        if !path_buf.exists() {
            return Ok(ToolResult::error(format!(
                "Database file not found: {}",
                path
            )));
        }

        let flags = if readonly {
            OpenFlags::SQLITE_OPEN_READ_ONLY
        } else {
            OpenFlags::SQLITE_OPEN_READ_WRITE
        };

        let conn = match Connection::open_with_flags(path, flags) {
            Ok(c) => c,
            Err(e) => {
                return Ok(ToolResult::error(format!("Failed to open database: {}", e)));
            }
        };

        let mut connections = self.connections.lock().await;
        connections.insert(alias.clone(), DbSession { conn, readonly });

        Ok(ToolResult::success(format!(
            "Connected to '{}' as alias '{}' (readonly: {})",
            path, alias, readonly
        ))
        .with_data(json!({ "alias": alias, "path": path, "readonly": readonly })))
    }

    async fn disconnect(&self, alias: &str) -> Result<ToolResult> {
        let mut connections = self.connections.lock().await;
        if connections.remove(alias).is_some() {
            Ok(ToolResult::success(format!(
                "Disconnected from '{}'",
                alias
            )))
        } else {
            Ok(ToolResult::error(format!(
                "No connection with alias '{}'",
                alias
            )))
        }
    }

    async fn query(&self, alias: &str, sql: &str) -> Result<ToolResult> {
        info!(alias = %alias, sql = %sql, "Executing query");

        let connections = self.connections.lock().await;
        let session = match connections.get(alias) {
            Some(c) => c,
            None => {
                return Ok(ToolResult::error(format!(
                    "No connection with alias '{}'. Use 'connect' first.",
                    alias
                )));
            }
        };
        let conn = &session.conn;

        // If readonly, we prevent write statements if we can detect them,
        // but opening with SQLITE_OPEN_READ_ONLY is the primary defense.
        // However, if the user opened as RW, but wants to be safe, we could check here too?
        // For now, rely on connection flags.

        let mut stmt = match conn.prepare(sql) {
            Ok(s) => s,
            Err(e) => {
                return Ok(ToolResult::error(format!("SQL error: {}", e)));
            }
        };

        let column_count = stmt.column_count();
        let column_names: Vec<String> = (0..column_count)
            .map(|i| stmt.column_name(i).unwrap_or("?").to_string())
            .collect();

        let mut rows_data: Vec<HashMap<String, Value>> = Vec::new();
        let mut rows_strings: Vec<Vec<String>> = Vec::new();

        let result = stmt.query_map([], |row| {
            let mut row_map = HashMap::new();
            let mut row_strs = Vec::new();

            for i in 0..column_count {
                let val = row.get_ref(i).ok();
                let (json_val, str_val) = match val {
                    Some(rusqlite::types::ValueRef::Null) => (Value::Null, "NULL".to_string()),
                    Some(rusqlite::types::ValueRef::Integer(n)) => (json!(n), n.to_string()),
                    Some(rusqlite::types::ValueRef::Real(f)) => (json!(f), format!("{:.6}", f)),
                    Some(rusqlite::types::ValueRef::Text(t)) => {
                        let s = String::from_utf8_lossy(t).to_string();
                        (json!(s), s)
                    }
                    Some(rusqlite::types::ValueRef::Blob(b)) => (
                        json!(format!("<blob:{} bytes>", b.len())),
                        format!("<blob:{} bytes>", b.len()),
                    ),
                    None => (Value::Null, "NULL".to_string()),
                };
                row_map.insert(column_names[i].clone(), json_val);
                row_strs.push(str_val);
            }
            Ok((row_map, row_strs))
        });

        match result {
            Ok(mapped_rows) => {
                for row_result in mapped_rows {
                    match row_result {
                        Ok((map, strs)) => {
                            rows_data.push(map);
                            rows_strings.push(strs);
                        }
                        Err(e) => {
                            return Ok(ToolResult::error(format!("Row error: {}", e)));
                        }
                    }
                }
            }
            Err(e) => {
                return Ok(ToolResult::error(format!("Query error: {}", e)));
            }
        }

        let row_count = rows_data.len();

        // Format as ASCII table
        let table_str = format_ascii_table(&column_names, &rows_strings);
        let output = format!("{}\n({} rows)", table_str, row_count);

        Ok(ToolResult::success(output).with_data(json!({
            "columns": column_names,
            "rows": rows_data,
            "row_count": row_count,
        })))
    }

    async fn list_tables(&self, alias: &str) -> Result<ToolResult> {
        info!(alias = %alias, "Listing tables");

        let connections = self.connections.lock().await;
        let session = match connections.get(alias) {
            Some(c) => c,
            None => {
                return Ok(ToolResult::error(format!(
                    "No connection with alias '{}'",
                    alias
                )));
            }
        };
        let conn = &session.conn;

        let mut stmt = conn.prepare(
            "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY name",
        )?;

        let tables: Vec<(String, String)> = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<Result<Vec<_>, _>>()?;

        let output = tables
            .iter()
            .map(|(name, typ)| format!("  {} ({})", name, typ))
            .collect::<Vec<_>>()
            .join("\n");

        let output = format!("Tables in '{}':\n{}", alias, output);

        Ok(ToolResult::success(output).with_data(json!({
            "tables": tables.iter().map(|(n, t)| json!({"name": n, "type": t})).collect::<Vec<_>>(),
        })))
    }

    async fn schema(&self, alias: &str, table: Option<&str>) -> Result<ToolResult> {
        info!(alias = %alias, table = ?table, "Inspecting schema");

        let connections = self.connections.lock().await;
        let session = match connections.get(alias) {
            Some(c) => c,
            None => {
                return Ok(ToolResult::error(format!(
                    "No connection with alias '{}'",
                    alias
                )));
            }
        };
        let conn = &session.conn;

        match table {
            Some(table_name) => {
                let mut stmt = conn.prepare(&format!("PRAGMA table_info({})", table_name))?;

                let columns: Vec<Value> = stmt
                    .query_map([], |row| {
                        Ok(json!({
                            "cid": row.get::<_, i64>(0)?,
                            "name": row.get::<_, String>(1)?,
                            "type": row.get::<_, String>(2)?,
                            "notnull": row.get::<_, i32>(3)? != 0,
                            "default": row.get::<_, Option<String>>(4)?,
                            "pk": row.get::<_, i32>(5)? != 0,
                        }))
                    })?
                    .collect::<Result<Vec<_>, _>>()?;

                if columns.is_empty() {
                    return Ok(ToolResult::error(format!(
                        "Table '{}' not found",
                        table_name
                    )));
                }

                let output = columns
                    .iter()
                    .map(|c| {
                        format!(
                            "  {} {} {}{}",
                            c["name"].as_str().unwrap_or("?"),
                            c["type"].as_str().unwrap_or("?"),
                            if c["pk"].as_bool().unwrap_or(false) {
                                "PK "
                            } else {
                                ""
                            },
                            if c["notnull"].as_bool().unwrap_or(false) {
                                "NOT NULL"
                            } else {
                                ""
                            }
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                let output = format!("Schema for '{}':\n{}", table_name, output);

                Ok(ToolResult::success(output).with_data(json!({
                    "table": table_name,
                    "columns": columns,
                })))
            }
            None => {
                // Show all table schemas
                let mut stmt = conn
                    .prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")?;

                let table_names: Vec<String> = stmt
                    .query_map([], |row| row.get(0))?
                    .collect::<Result<Vec<_>, _>>()?;

                let mut all_schemas = Vec::new();
                let mut output_parts = Vec::new();

                for tbl in &table_names {
                    let mut col_stmt = conn.prepare(&format!("PRAGMA table_info({})", tbl))?;
                    let columns: Vec<Value> = col_stmt
                        .query_map([], |row| {
                            Ok(json!({
                                "name": row.get::<_, String>(1)?,
                                "type": row.get::<_, String>(2)?,
                                "pk": row.get::<_, i32>(5)? != 0,
                            }))
                        })?
                        .collect::<Result<Vec<_>, _>>()?;

                    let cols_str = columns
                        .iter()
                        .map(|c| {
                            format!(
                                "    {} {}{}",
                                c["name"].as_str().unwrap_or("?"),
                                c["type"].as_str().unwrap_or("?"),
                                if c["pk"].as_bool().unwrap_or(false) {
                                    " PK"
                                } else {
                                    ""
                                }
                            )
                        })
                        .collect::<Vec<_>>()
                        .join("\n");

                    output_parts.push(format!("  {}:\n{}", tbl, cols_str));
                    all_schemas.push(json!({
                        "table": tbl,
                        "columns": columns,
                    }));
                }

                let output = format!("Database schema:\n{}", output_parts.join("\n\n"));

                Ok(ToolResult::success(output).with_data(json!({
                    "tables": all_schemas,
                })))
            }
        }
    }

    async fn explain(&self, alias: &str, sql: &str) -> Result<ToolResult> {
        info!(alias = %alias, sql = %sql, "Explaining query");

        let connections = self.connections.lock().await;
        let session = match connections.get(alias) {
            Some(c) => c,
            None => {
                return Ok(ToolResult::error(format!(
                    "No connection with alias '{}'",
                    alias
                )));
            }
        };
        let conn = &session.conn;

        let explain_sql = format!("EXPLAIN QUERY PLAN {}", sql);
        let mut stmt = match conn.prepare(&explain_sql) {
            Ok(s) => s,
            Err(e) => {
                return Ok(ToolResult::error(format!("SQL error: {}", e)));
            }
        };

        let plans: Vec<Value> = stmt
            .query_map([], |row| {
                Ok(json!({
                    "id": row.get::<_, i64>(0)?,
                    "parent": row.get::<_, i64>(1)?,
                    "notused": row.get::<_, i64>(2)?,
                    "detail": row.get::<_, String>(3)?,
                }))
            })?
            .collect::<Result<Vec<_>, _>>()?;

        let output = plans
            .iter()
            .map(|p| format!("  {}", p["detail"].as_str().unwrap_or("?")))
            .collect::<Vec<_>>()
            .join("\n");

        let output = format!("Query plan for:\n  {}\n\n{}", sql, output);

        Ok(ToolResult::success(output).with_data(json!({
            "sql": sql,
            "plan": plans,
        })))
    }
}

#[async_trait]
impl Tool for DatabaseTool {
    fn name(&self) -> &str {
        "database"
    }

    fn description(&self) -> &str {
        "Direct SQL access to SQLite databases. Actions: connect, query, schema, explain, list_tables, disconnect. Read-only by default; write queries require approval."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "required": ["action"],
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["connect", "query", "schema", "explain", "list_tables", "disconnect"],
                    "description": "Database action to perform"
                },
                "path": {
                    "type": "string",
                    "description": "Path to SQLite database file (required for 'connect')"
                },
                "alias": {
                    "type": "string",
                    "description": "Connection alias (defaults to filename)"
                },
                "readonly": {
                    "type": "boolean",
                    "description": "Open in read-only mode (default: true). Set to false for writes."
                },
                "sql": {
                    "type": "string",
                    "description": "SQL query (required for 'query' and 'explain')"
                },
                "table": {
                    "type": "string",
                    "description": "Table name (optional for 'schema' to show single table)"
                }
            }
        })
    }

    async fn execute(&self, params: Value, _context: ToolContext) -> Result<ToolResult> {
        let action = params
            .get("action")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'action' parameter"))?;

        match action {
            "connect" => {
                let path = params
                    .get("path")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing 'path' for connect"))?;
                let alias = params.get("alias").and_then(|v| v.as_str());
                let readonly = params
                    .get("readonly")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true);
                self.connect(path, alias, readonly).await
            }
            "query" => {
                let alias = params
                    .get("alias")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing 'alias' for query"))?;
                let sql = params
                    .get("sql")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing 'sql' for query"))?;
                self.query(alias, sql).await
            }
            "schema" => {
                let alias = params
                    .get("alias")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing 'alias' for schema"))?;
                let table = params.get("table").and_then(|v| v.as_str());
                self.schema(alias, table).await
            }

            "explain" => {
                let alias = params
                    .get("alias")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing 'alias' for explain"))?;
                let sql = params
                    .get("sql")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing 'sql' for explain"))?;
                self.explain(alias, sql).await
            }
            "list_tables" => {
                let alias = params
                    .get("alias")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing 'alias' for list_tables"))?;
                self.list_tables(alias).await
            }
            "disconnect" => {
                let alias = params
                    .get("alias")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing 'alias' for disconnect"))?;
                self.disconnect(alias).await
            }
            _ => Ok(ToolResult::error(format!("Unknown action: {}", action))),
        }
    }

    fn permission_tier(&self) -> PermissionTier {
        PermissionTier::Medium
    }
}

/// Format rows as an ASCII table.
fn format_ascii_table(columns: &[String], rows: &[Vec<String>]) -> String {
    if columns.is_empty() {
        return "(no columns)".to_string();
    }

    // Calculate column widths
    let mut widths: Vec<usize> = columns.iter().map(|c| c.len()).collect();
    for row in rows {
        for (i, cell) in row.iter().enumerate() {
            if i < widths.len() && cell.len() > widths[i] {
                widths[i] = cell.len();
            }
        }
    }

    // Cap column widths at 50 characters
    for w in widths.iter_mut() {
        if *w > 50 {
            *w = 50;
        }
    }

    let mut output = String::new();

    // Header row
    let header: Vec<String> = columns
        .iter()
        .enumerate()
        .map(|(i, c)| format!("{:width$}", c, width = widths[i]))
        .collect();
    output.push_str("| ");
    output.push_str(&header.join(" | "));
    output.push_str(" |\n");

    // Separator
    let sep: Vec<String> = widths.iter().map(|w| "-".repeat(*w)).collect();
    output.push_str("|-");
    output.push_str(&sep.join("-|-"));
    output.push_str("-|\n");

    // Data rows
    for row in rows {
        let cells: Vec<String> = row
            .iter()
            .enumerate()
            .map(|(i, cell)| {
                let width = widths.get(i).copied().unwrap_or(10);
                let truncated = if cell.len() > width {
                    format!("{}...", &cell[..width.saturating_sub(3)])
                } else {
                    cell.clone()
                };
                format!("{:width$}", truncated, width = width)
            })
            .collect();
        output.push_str("| ");
        output.push_str(&cells.join(" | "));
        output.push_str(" |\n");
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[tokio::test]
    async fn test_database_readonly() -> Result<()> {
        let db_tool = DatabaseTool::new();
        let db_path = std::env::temp_dir().join("test_db_readonly.sqlite");
        let db_str = db_path.to_str().unwrap();

        // 1. Create DB and Table in RW mode (using direct connection or tool)
        {
            let conn = Connection::open(&db_path)?;
            conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, val TEXT)", [])?;
            conn.execute("INSERT INTO test (val) VALUES ('foo')", [])?;
        }

        // 2. Connect ReadOnly via Tool (default)
        let res = db_tool.connect(db_str, Some("ro"), true).await?;
        assert!(res.success);

        // 3. Query (SELECT) - Should succeed
        let res = db_tool.query("ro", "SELECT * FROM test").await?;
        assert!(res.success);
        assert!(res.output.contains("foo"));

        // 4. Query (INSERT) - Should fail
        let res = db_tool
            .query("ro", "INSERT INTO test (val) VALUES ('bar')")
            .await;
        // SQLite read-only constraint violation usually returns generic SQLite error
        assert!(res.is_ok()); // The tool returns Ok(ToolResult::error(...))
        let tool_res = res.unwrap();
        assert!(!tool_res.success);
        let error_msg = tool_res.error.unwrap_or_default();
        assert!(error_msg.contains("attempt to write a readonly database"));

        // 5. Connect RW via Tool
        db_tool.disconnect("ro").await?;
        let res = db_tool.connect(db_str, Some("rw"), false).await?;
        assert!(res.success);

        // 6. Query (INSERT) - Should succeed
        let res = db_tool
            .query("rw", "INSERT INTO test (val) VALUES ('bar')")
            .await?;
        assert!(res.success);

        // Cleanup
        let _ = fs::remove_file(db_path);

        Ok(())
    }
}
