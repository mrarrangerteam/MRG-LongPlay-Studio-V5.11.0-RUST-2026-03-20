# MRARRANGER Skills for Cursor IDE

นำ Skill ทั้ง 26 ตัวจาก Cowork มาใช้ใน Cursor ได้เหมือนกัน!

## 🏗️ Architecture

```
mrarranger-cursor-skills/
├── .cursorrules              ← Master rules (copy ไปโปรเจกต์ที่ต้องการ)
├── .cursor/
│   ├── mcp.json              ← Cursor MCP config (per-project)
│   └── rules/
│       └── mrarranger-system.mdc  ← Auto-routing rules
├── mcp-server/
│   ├── index.js              ← MCP Server (serve skills เป็น tools)
│   └── package.json
├── skills/                   ← Skills ทั้งหมด (copy จาก Cowork)
│   ├── mrarranger-dev-tools/
│   │   ├── SKILL.md
│   │   └── references/
│   ├── mrarranger-branding-copy/
│   │   ├── SKILL.md
│   │   └── references/
│   └── ... (26 skills)
└── scripts/
    └── setup.sh              ← Auto setup script
```

## ⚡ Quick Setup

### Option A: Auto Setup
```bash
# Clone/copy โฟลเดอร์นี้
cd mrarranger-cursor-skills

# Run setup (ระบุ path ไปยัง Cowork skills)
bash scripts/setup.sh /path/to/your/cowork/skills
```

### Option B: Manual Setup

**Step 1: Copy Skills**
```bash
# Copy skills จาก Cowork มาไว้ใน skills/
mkdir -p skills
cp -r /path/to/cowork/skills/mrarranger-* skills/
cp -r /path/to/cowork/skills/openclaw-full-suite skills/
```

**Step 2: Install MCP Server**
```bash
cd mcp-server
npm install
```

**Step 3: Configure Cursor MCP**

เพิ่มใน `~/.cursor/mcp.json` (global) หรือ `.cursor/mcp.json` (per-project):
```json
{
  "mcpServers": {
    "mrarranger-skills": {
      "command": "node",
      "args": ["/FULL/PATH/TO/mrarranger-cursor-skills/mcp-server/index.js"],
      "env": {
        "MRARRANGER_SKILLS_DIR": "/FULL/PATH/TO/mrarranger-cursor-skills/skills"
      }
    }
  }
}
```

**Step 4: Copy .cursorrules**
```bash
# Copy ไปโปรเจกต์ที่ต้องการใช้
cp .cursorrules /path/to/your/project/
```

**Step 5: Restart Cursor**

## 🎯 How to Use

### MCP Tools ที่ใช้ได้ใน Cursor

| Tool | Description |
|------|-------------|
| `list_skills` | ดู Skills ทั้งหมด (filter by category) |
| `load_skill` | โหลด SKILL.md instructions เต็ม |
| `load_reference` | โหลด reference file เฉพาะ sub-skill |
| `list_references` | ดู reference files ของ skill |
| `search_skills` | ค้นหา skill จาก keyword |

### ตัวอย่างการใช้

```
User: review โค้ดนี้ให้หน่อย
Cursor → search_skills("code review") → load_skill("mrarranger-dev-tools")
       → load_reference("mrarranger-dev-tools", "code-review-graph.md")
       → Execute ตาม instructions

User: /brand สร้าง brand strategy
Cursor → load_skill("mrarranger-branding-copy")
       → load_reference("mrarranger-branding-copy", "alphamind-brand-cortex.md")
       → Execute

User: สร้าง pitch deck
Cursor → search_skills("pitch") → load_skill("mrarranger-investor-fundraise")
       → Execute
```

## 📋 Skills ทั้งหมด (26 Skills)

### Development & Engineering (7)
- **dev-fullstack** — React, Next.js, Auth, API, Supabase, CMS, SEO
- **dev-tools** — Code review, Debug, CLI, MCP Builder, Security audit
- **creative-dev** — Mobile, Game dev, Audio plugin, DAW, Video editor
- **python-backend** — Django, FastAPI, PostgreSQL, Redis, Data science
- **desktop-iot** — Electron, Tauri, Arduino, ESP32, WebRTC, Go, Rust
- **devops-ai** — Docker, K8s, CI/CD, ML, RAG, Web3
- **cybersecurity** — Pentest, OWASP, SOC, Forensics, Zero Trust

### Business & Strategy (6)
- **business-system** — Proposals, Legal, Research, Automation
- **project-mgmt** — Agile, Scrum, Kanban, Roadmap, RACI
- **investor-fundraise** — Pitch deck, Term sheet, Cap table
- **marketing-sales** — Funnels, Ads, E-commerce, SaaS, Analytics
- **hr-team** — Hiring, Onboarding, OKR, Performance review
- **customer-cx** — Support, Ticketing, NPS, CRM

### Content & Branding (4)
- **branding-copy** — Brand strategy, Copywriting, Funnel scripts
- **content-social** — Viral content, Social media, Blog, Newsletter
- **i18n-translate** — Translation, Localization, i18n, RTL
- **music-visual** — Music production, Suno, Design, UI/UX

### Specialized (3)
- **trading-suite** — Polymarket, prediction market, backtesting
- **asteroid-astrology** — Natal chart, transit, synastry
- **openclaw-full-suite** — Multi-agent, Channels, MCP tools

## 🔧 Troubleshooting

**MCP Server ไม่ทำงาน:**
```bash
# ทดสอบ manual
cd mcp-server
node index.js
# ควรเห็น "MRARRANGER Skills MCP Server running on stdio"
```

**Skills ไม่เจอ:**
```bash
# ตรวจสอบว่า skills/ มีไฟล์
ls skills/mrarranger-*/SKILL.md
```

**Cursor ไม่เห็น MCP:**
- ตรวจสอบ `.cursor/mcp.json` ว่า path ถูกต้อง
- Restart Cursor หลัง config
- ใช้ absolute path (ไม่ใช้ ~ หรือ relative path)
