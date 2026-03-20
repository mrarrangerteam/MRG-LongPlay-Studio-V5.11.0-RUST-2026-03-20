#!/bin/bash
# ==========================================
# MRARRANGER Skills - Cursor Setup Script
# ==========================================
# Usage: bash scripts/setup.sh [SKILLS_SOURCE_DIR]
#
# SKILLS_SOURCE_DIR = path to your Cowork skills folder
# Default: searches common locations

set -e

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔═══════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  MRARRANGER Skills → Cursor Setup         ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════╝${NC}"
echo ""

# ─── Find project root ───
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SKILLS_DEST="$PROJECT_DIR/skills"

# ─── Find source skills ───
SKILLS_SOURCE="${1:-}"

if [ -z "$SKILLS_SOURCE" ]; then
  echo -e "${YELLOW}Searching for Cowork skills directory...${NC}"

  # Common locations for Cowork skills
  SEARCH_PATHS=(
    "$HOME/Library/Application Support/Claude/skills"
    "$HOME/.config/claude/skills"
    "$HOME/.claude/skills"
    "$HOME/Documents/Claude/skills"
  )

  for p in "${SEARCH_PATHS[@]}"; do
    if [ -d "$p" ] && ls "$p"/mrarranger-*/SKILL.md &>/dev/null 2>&1; then
      SKILLS_SOURCE="$p"
      echo -e "${GREEN}Found skills at: $p${NC}"
      break
    fi
  done
fi

if [ -z "$SKILLS_SOURCE" ]; then
  echo -e "${RED}Could not find Cowork skills directory automatically.${NC}"
  echo ""
  echo "Please provide the path manually:"
  echo "  bash scripts/setup.sh /path/to/your/skills"
  echo ""
  echo "Tip: In Cowork, your skills are usually at:"
  echo "  ~/Library/Application Support/Claude/skills"
  echo ""
  echo -e "${YELLOW}Or copy them manually:${NC}"
  echo "  mkdir -p $SKILLS_DEST"
  echo "  cp -r /path/to/skills/mrarranger-* $SKILLS_DEST/"
  echo "  cp -r /path/to/skills/openclaw-full-suite $SKILLS_DEST/"
  exit 1
fi

# ─── Copy skills ───
echo -e "\n${BLUE}Step 1: Copying skills...${NC}"
mkdir -p "$SKILLS_DEST"

count=0
for skill_dir in "$SKILLS_SOURCE"/mrarranger-* "$SKILLS_SOURCE"/openclaw-full-suite; do
  if [ -d "$skill_dir" ]; then
    skill_name=$(basename "$skill_dir")
    echo "  📦 $skill_name"
    cp -r "$skill_dir" "$SKILLS_DEST/"
    count=$((count + 1))
  fi
done

echo -e "${GREEN}  ✅ Copied $count skills${NC}"

# ─── Install MCP server dependencies ───
echo -e "\n${BLUE}Step 2: Installing MCP server dependencies...${NC}"
cd "$PROJECT_DIR/mcp-server"
npm install
echo -e "${GREEN}  ✅ Dependencies installed${NC}"

# ─── Setup .cursor/mcp.json in user's home (global) ───
echo -e "\n${BLUE}Step 3: Configuring Cursor MCP...${NC}"

CURSOR_MCP_DIR="$HOME/.cursor"
CURSOR_MCP_FILE="$CURSOR_MCP_DIR/mcp.json"

mkdir -p "$CURSOR_MCP_DIR"

# Build the MCP config
MCP_CONFIG="{
  \"mcpServers\": {
    \"mrarranger-skills\": {
      \"command\": \"node\",
      \"args\": [\"$PROJECT_DIR/mcp-server/index.js\"],
      \"env\": {
        \"MRARRANGER_SKILLS_DIR\": \"$SKILLS_DEST\"
      }
    }
  }
}"

if [ -f "$CURSOR_MCP_FILE" ]; then
  echo -e "${YELLOW}  ⚠️  $CURSOR_MCP_FILE already exists${NC}"
  echo "  Merging configuration..."

  # Check if node/npx is available for JSON merge
  if command -v node &>/dev/null; then
    node -e "
      const fs = require('fs');
      const existing = JSON.parse(fs.readFileSync('$CURSOR_MCP_FILE', 'utf-8'));
      const newConfig = $MCP_CONFIG;
      existing.mcpServers = { ...existing.mcpServers, ...newConfig.mcpServers };
      fs.writeFileSync('$CURSOR_MCP_FILE', JSON.stringify(existing, null, 2));
      console.log('  Merged successfully');
    "
  else
    echo "  Cannot auto-merge. Please manually add to $CURSOR_MCP_FILE:"
    echo "$MCP_CONFIG"
  fi
else
  echo "$MCP_CONFIG" > "$CURSOR_MCP_FILE"
  echo -e "${GREEN}  ✅ Created $CURSOR_MCP_FILE${NC}"
fi

# ─── Copy .cursorrules to project ───
echo -e "\n${BLUE}Step 4: .cursorrules is ready at project root${NC}"
echo -e "${GREEN}  ✅ Copy .cursorrules to any project you want to use skills in${NC}"

# ─── Done ───
echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  ✅ Setup Complete!                        ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════╝${NC}"
echo ""
echo -e "Next steps:"
echo -e "  1. ${BLUE}Restart Cursor${NC} to load the MCP server"
echo -e "  2. Copy ${BLUE}.cursorrules${NC} to your project root"
echo -e "  3. Try asking: ${YELLOW}/code review my project${NC}"
echo -e "  4. Or just ask naturally and skills auto-route!"
echo ""
echo -e "MCP Server: $PROJECT_DIR/mcp-server/index.js"
echo -e "Skills Dir: $SKILLS_DEST"
echo -e "Cursor MCP: $CURSOR_MCP_FILE"
