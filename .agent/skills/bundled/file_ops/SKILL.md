---
name: File Operations
description: Common file and directory operations like find, search, copy, and organize
version: 1.0.0
---

# File Operations Skill

Provides patterns for common file system tasks.

## Usage

### Find Files by Name

```bash
find /path -name "*.ext"
```

### Search File Contents

```bash
grep -r "pattern" /path
```

### Count Lines in Files

```bash
wc -l file.txt
# Or for multiple files
find . -name "*.py" | xargs wc -l
```

### List Large Files

```bash
find . -type f -size +100M -exec ls -lh {} \;
```

### Recent Files

```bash
find . -type f -mtime -7 -name "*.py"
```

### Directory Size

```bash
du -sh /path/to/dir
```

### Safe Delete

```bash
# Move to trash instead of permanent delete
mv file ~/.local/share/Trash/files/
```

## Examples

- "Find all Python files in this project"
- "Search for TODO comments"
- "What are the largest files here?"
- "Show me files modified today"
