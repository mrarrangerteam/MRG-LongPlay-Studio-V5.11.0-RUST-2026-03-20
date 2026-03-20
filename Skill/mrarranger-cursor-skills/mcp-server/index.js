#!/usr/bin/env node

/**
 * MRARRANGER Skills MCP Server
 * ============================
 * Serves 26 specialized AI skills to Cursor IDE via Model Context Protocol.
 *
 * Tools provided:
 * - list_skills: List all available skills with descriptions
 * - load_skill: Load a specific skill's full instructions (SKILL.md)
 * - load_reference: Load a specific reference file from a skill
 * - search_skills: Search skills by keyword/trigger
 * - list_references: List all reference files for a skill
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import fs from "fs";
import path from "path";

// Skills directory - configurable via env var
const SKILLS_DIR = process.env.MRARRANGER_SKILLS_DIR || path.join(process.env.HOME || "", ".mrarranger-skills");

// ─────────────────────────────────────────────
// Skill Registry
// ─────────────────────────────────────────────

const SKILL_REGISTRY = {
  "mrarranger-dev-fullstack": {
    name: "Dev Fullstack",
    description: "React+Next.js, Auth, API (REST/GraphQL/tRPC), Supabase, CMS, SEO, UI System, Remotion",
    commands: ["/react", "/nextjs", "/auth", "/api", "/graphql", "/supabase", "/cms", "/frontend", "/ui", "/seo"],
    triggers: ["React", "Next.js", "Auth", "OAuth", "JWT", "API", "GraphQL", "Supabase", "CMS", "SEO", "Tailwind", "UI system"]
  },
  "mrarranger-dev-tools": {
    name: "Dev Tools",
    description: "Code Master, Debug Master, CLI Tools, MCP Builder, Technical Docs, Vibe Code Guardian, Code Review Graph",
    commands: ["/code", "/debug", "/cli", "/mcp", "/docs", "/guard", "/review-pr", "/review-delta", "/blast", "/security-review"],
    triggers: ["code review", "PR review", "debugging", "MCP", "clean code", "SOLID", "security audit", "OWASP"]
  },
  "mrarranger-creative-dev": {
    name: "Creative Dev",
    description: "Mobile Dev (React Native/Flutter), Game Dev (Unity/Godot), Audio Plugin (VST3/JUCE), DAW Engine, Video Editor",
    commands: ["/mobile", "/flutter", "/react-native", "/game", "/unity", "/godot", "/vst", "/daw", "/video-editor"],
    triggers: ["mobile app", "Flutter", "React Native", "game dev", "Unity", "Godot", "VST", "audio plugin", "DAW"]
  },
  "mrarranger-python-backend": {
    name: "Python Backend & Data",
    description: "Django, FastAPI, Flask, PostgreSQL, MongoDB, Redis, Stripe, Data Science, GDPR/PDPA, Kafka",
    commands: ["/python", "/django", "/fastapi", "/flask", "/postgres", "/mongodb", "/redis", "/stripe", "/datascience", "/kafka"],
    triggers: ["Python", "Django", "FastAPI", "Flask", "PostgreSQL", "MongoDB", "Redis", "Stripe", "data science", "Kafka"]
  },
  "mrarranger-desktop-iot": {
    name: "Desktop, IoT & Systems",
    description: "Electron, Tauri, Arduino, ESP32, MQTT, WebRTC, Go, Rust, Bluetooth, Hardware Integration",
    commands: ["/electron", "/tauri", "/arduino", "/esp32", "/iot", "/mqtt", "/webrtc", "/go", "/rust"],
    triggers: ["Electron", "Tauri", "desktop app", "Arduino", "ESP32", "IoT", "MQTT", "WebRTC", "Go", "Rust"]
  },
  "mrarranger-devops-ai": {
    name: "DevOps + AI",
    description: "Docker, K8s, CI/CD, AWS/GCP/Azure, Testing, ETL/Pipeline, Web3, AI ImageGen, ML/RAG",
    commands: ["/devops", "/docker", "/k8s", "/cicd", "/test", "/etl", "/web3", "/imagegen", "/ml", "/rag"],
    triggers: ["Docker", "Kubernetes", "CI/CD", "AWS", "ML", "RAG", "Web3", "Solidity", "AI image"]
  },
  "mrarranger-cybersecurity": {
    name: "Cybersecurity",
    description: "Penetration Testing, OWASP, SOC, Threat Modeling, Incident Response, Forensics, Zero Trust",
    commands: ["/pentest", "/owasp", "/threat", "/soc", "/incident", "/forensics", "/zerotrust"],
    triggers: ["cybersecurity", "pentest", "OWASP", "SOC", "threat model", "incident response", "forensics"]
  },
  "mrarranger-business-system": {
    name: "Business & System",
    description: "Business docs, Legal, Course Creation, Research, Data Analysis, Automation, Prompt Engineering",
    commands: ["/business", "/legal", "/course", "/research", "/data", "/life", "/persona", "/brain", "/logic", "/prompt", "/qc", "/auto"],
    triggers: ["business proposal", "legal", "course creation", "research", "data analysis", "automation", "prompt engineering"]
  },
  "mrarranger-project-mgmt": {
    name: "Project Management",
    description: "Agile/Scrum, Sprint Planning, Kanban, Backlog, User Stories, Roadmap, RACI, Risk Management",
    commands: ["/sprint", "/scrum", "/kanban", "/backlog", "/userstory", "/estimate", "/roadmap", "/risk", "/raci"],
    triggers: ["agile", "scrum", "sprint", "kanban", "backlog", "roadmap", "project management", "RACI"]
  },
  "mrarranger-investor-fundraise": {
    name: "Investor & Fundraising",
    description: "Pitch Deck, Term Sheet, Cap Table, Valuation, Due Diligence, SAFE, Funding Rounds",
    commands: ["/pitch", "/investor", "/termsheet", "/captable", "/valuation", "/fundraise"],
    triggers: ["pitch deck", "investor", "fundraising", "term sheet", "cap table", "valuation", "SAFE", "Series A"]
  },
  "mrarranger-marketing-sales": {
    name: "Marketing & Sales",
    description: "Sales Funnels, Ads, Outreach, E-commerce, SaaS, Analytics, Finance",
    commands: ["/sales", "/funnel", "/ads", "/outreach", "/ecommerce", "/saas", "/analytics", "/finance"],
    triggers: ["sales funnel", "marketing", "ads", "e-commerce", "SaaS", "analytics", "finance"]
  },
  "mrarranger-hr-team": {
    name: "HR & Team Management",
    description: "Hiring, Onboarding, OKR, Performance Review, Team Building, Employee Handbook, Leadership",
    commands: ["/hire", "/onboard", "/okr", "/team", "/handbook", "/compensation", "/leadership"],
    triggers: ["hiring", "onboarding", "HR", "OKR", "performance review", "team building", "handbook"]
  },
  "mrarranger-customer-cx": {
    name: "Customer Experience",
    description: "Support Strategy, Ticketing, Chatbot, NPS/CSAT, Customer Journey, SLA, Retention, CRM",
    commands: ["/support", "/ticket", "/chatbot", "/nps", "/journey", "/sla", "/retention", "/crm"],
    triggers: ["customer support", "ticketing", "chatbot", "NPS", "customer journey", "SLA", "CRM"]
  },
  "mrarranger-branding-copy": {
    name: "Branding & Copywriting",
    description: "Brand Strategy, Archetypes, Sales Copy, Funnel Scripts, Webinar Scripts, Story Selling, Voice Cloning",
    commands: ["/brand", "/copy", "/cortex", "/archetype", "/funnel", "/webinar", "/story", "/clone"],
    triggers: ["branding", "copywriting", "sales page", "funnel script", "webinar", "archetypes", "headline"]
  },
  "mrarranger-content-social": {
    name: "Content & Social Media",
    description: "Viral Content, Social Media (IG/TikTok/LinkedIn), Blog, Newsletter, Content Strategy, Thai Copywriting",
    commands: ["/viral", "/hook", "/social", "/content", "/blog", "/newsletter", "/mass", "/thai"],
    triggers: ["content creation", "social media", "viral", "blog", "newsletter", "Thai copywriting"]
  },
  "mrarranger-i18n-translate": {
    name: "Internationalization & Translation",
    description: "Multi-language Translation, i18n Implementation, RTL, Locale Management, SEO Multilingual",
    commands: ["/translate", "/localize", "/i18n", "/rtl", "/hreflang"],
    triggers: ["translation", "localization", "i18n", "multilingual", "RTL", "hreflang"]
  },
  "mrarranger-music-visual": {
    name: "Music & Visual Production",
    description: "Music Production, Suno AI Songs, Video/Image, Design, UI/UX, DSP Audio Effects",
    commands: ["/music", "/suno", "/visual", "/design", "/ui", "/dsp"],
    triggers: ["music", "Suno", "video production", "design", "UI/UX", "DSP", "audio effects"]
  },
  "mrarranger-trading-suite": {
    name: "Trading Suite",
    description: "Prediction Markets (Polymarket/Kalshi), Edge Analysis, Backtesting, Signal Detection, Risk Management",
    commands: ["/edge", "/backtest", "/signal", "/risk", "/whale", "/onchain"],
    triggers: ["polymarket", "prediction market", "trading", "backtest", "signal", "whale tracking"]
  },
  "mrarranger-asteroid-astrology": {
    name: "Asteroid Astrology",
    description: "Natal Chart + Asteroid Interpretation + Transit + Synastry + Composite + Return Chart",
    commands: ["/natal", "/transit", "/synastry", "/zodiac"],
    triggers: ["natal chart", "horoscope", "zodiac", "asteroid", "transit", "synastry", "ราศี", "ดวงดาว"]
  },
  "openclaw-full-suite": {
    name: "OpenClaw Full Suite",
    description: "Multi-Agent Routing, Automation, Channels (WhatsApp/Telegram/Slack), MCP Tools, Skills Dev",
    commands: ["/claw-agent", "/claw-auto", "/claw-channel", "/claw-hub", "/claw-mcp", "/claw-skill"],
    triggers: ["OpenClaw", "multi-agent", "chatbot channels", "WhatsApp bot", "Telegram bot"]
  }
};

// ─────────────────────────────────────────────
// File helpers
// ─────────────────────────────────────────────

function readSkillFile(skillName) {
  const filePath = path.join(SKILLS_DIR, skillName, "SKILL.md");
  if (!fs.existsSync(filePath)) {
    throw new Error(`Skill '${skillName}' not found at ${filePath}`);
  }
  return fs.readFileSync(filePath, "utf-8");
}

function readReferenceFile(skillName, refName) {
  // Try with and without 'references/' prefix
  let filePath = path.join(SKILLS_DIR, skillName, "references", refName);
  if (!fs.existsSync(filePath)) {
    filePath = path.join(SKILLS_DIR, skillName, refName);
  }
  if (!fs.existsSync(filePath)) {
    throw new Error(`Reference '${refName}' not found in skill '${skillName}'`);
  }
  return fs.readFileSync(filePath, "utf-8");
}

function listReferenceFiles(skillName) {
  const refsDir = path.join(SKILLS_DIR, skillName, "references");
  if (!fs.existsSync(refsDir)) {
    return [];
  }
  return fs.readdirSync(refsDir).filter(f => f.endsWith(".md"));
}

function listScriptFiles(skillName) {
  const scriptsDir = path.join(SKILLS_DIR, skillName, "scripts");
  if (!fs.existsSync(scriptsDir)) {
    return [];
  }
  return fs.readdirSync(scriptsDir);
}

// ─────────────────────────────────────────────
// MCP Server Setup
// ─────────────────────────────────────────────

const server = new Server(
  { name: "mrarranger-skills", version: "1.0.0" },
  { capabilities: { tools: {} } }
);

// ─── List Tools ───
server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "list_skills",
      description: "List all available MRARRANGER skills with their descriptions, commands, and triggers. Use this first to discover what skills are available.",
      inputSchema: {
        type: "object",
        properties: {
          category: {
            type: "string",
            description: "Optional filter: 'dev', 'business', 'content', 'specialized', or 'all'",
            enum: ["dev", "business", "content", "specialized", "all"]
          }
        }
      }
    },
    {
      name: "load_skill",
      description: "Load a specific skill's full SKILL.md instructions. Use this to get the complete skill instructions including routing tables and workflow guides.",
      inputSchema: {
        type: "object",
        properties: {
          skill_name: {
            type: "string",
            description: "The skill identifier (e.g., 'mrarranger-dev-tools', 'mrarranger-branding-copy')"
          }
        },
        required: ["skill_name"]
      }
    },
    {
      name: "load_reference",
      description: "Load a specific reference/sub-skill file from a skill. Reference files contain detailed instructions for specific capabilities within a skill.",
      inputSchema: {
        type: "object",
        properties: {
          skill_name: {
            type: "string",
            description: "The skill identifier (e.g., 'mrarranger-dev-tools')"
          },
          reference_name: {
            type: "string",
            description: "The reference filename (e.g., 'code-review-graph.md', 'inception-copywriting.md')"
          }
        },
        required: ["skill_name", "reference_name"]
      }
    },
    {
      name: "list_references",
      description: "List all reference files available for a specific skill. Use this to see what sub-skills and detailed guides are available.",
      inputSchema: {
        type: "object",
        properties: {
          skill_name: {
            type: "string",
            description: "The skill identifier"
          }
        },
        required: ["skill_name"]
      }
    },
    {
      name: "search_skills",
      description: "Search for skills by keyword. Matches against skill names, descriptions, commands, and triggers.",
      inputSchema: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description: "Search keyword (e.g., 'React', 'marketing', 'docker', 'ดูดวง')"
          }
        },
        required: ["query"]
      }
    }
  ]
}));

// ─── Call Tool Handler ───
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    switch (name) {
      case "list_skills": {
        const category = args?.category || "all";
        const categories = {
          dev: ["mrarranger-dev-fullstack", "mrarranger-dev-tools", "mrarranger-creative-dev", "mrarranger-python-backend", "mrarranger-desktop-iot", "mrarranger-devops-ai", "mrarranger-cybersecurity"],
          business: ["mrarranger-business-system", "mrarranger-project-mgmt", "mrarranger-investor-fundraise", "mrarranger-marketing-sales", "mrarranger-hr-team", "mrarranger-customer-cx"],
          content: ["mrarranger-branding-copy", "mrarranger-content-social", "mrarranger-i18n-translate", "mrarranger-music-visual"],
          specialized: ["mrarranger-trading-suite", "mrarranger-asteroid-astrology", "openclaw-full-suite"]
        };

        let skillIds = category === "all"
          ? Object.keys(SKILL_REGISTRY)
          : (categories[category] || Object.keys(SKILL_REGISTRY));

        const result = skillIds.map(id => {
          const s = SKILL_REGISTRY[id];
          const refs = listReferenceFiles(id);
          return `## ${s.name} (\`${id}\`)\n${s.description}\n**Commands:** ${s.commands.join(", ")}\n**Triggers:** ${s.triggers.join(", ")}\n**References:** ${refs.length > 0 ? refs.join(", ") : "none"}`;
        }).join("\n\n---\n\n");

        return { content: [{ type: "text", text: `# MRARRANGER Skills (${category})\n\n${result}` }] };
      }

      case "load_skill": {
        const content = readSkillFile(args.skill_name);
        return { content: [{ type: "text", text: content }] };
      }

      case "load_reference": {
        const content = readReferenceFile(args.skill_name, args.reference_name);
        return { content: [{ type: "text", text: content }] };
      }

      case "list_references": {
        const refs = listReferenceFiles(args.skill_name);
        const scripts = listScriptFiles(args.skill_name);
        let result = `# References for ${args.skill_name}\n\n`;
        if (refs.length > 0) {
          result += `## Reference Files\n${refs.map(r => `- ${r}`).join("\n")}\n\n`;
        }
        if (scripts.length > 0) {
          result += `## Script Files\n${scripts.map(s => `- ${s}`).join("\n")}\n\n`;
        }
        if (refs.length === 0 && scripts.length === 0) {
          result += "No reference or script files found. The skill uses only SKILL.md.";
        }
        return { content: [{ type: "text", text: result }] };
      }

      case "search_skills": {
        const query = (args.query || "").toLowerCase();
        const matches = [];

        for (const [id, skill] of Object.entries(SKILL_REGISTRY)) {
          const searchText = [
            id, skill.name, skill.description,
            ...skill.commands, ...skill.triggers
          ].join(" ").toLowerCase();

          if (searchText.includes(query)) {
            const refs = listReferenceFiles(id);
            matches.push({
              id,
              name: skill.name,
              description: skill.description,
              commands: skill.commands,
              references: refs
            });
          }
        }

        if (matches.length === 0) {
          return { content: [{ type: "text", text: `No skills found matching "${args.query}". Try broader terms.` }] };
        }

        const result = matches.map(m =>
          `### ${m.name} (\`${m.id}\`)\n${m.description}\nCommands: ${m.commands.join(", ")}\nReferences: ${m.references.length > 0 ? m.references.join(", ") : "none"}`
        ).join("\n\n");

        return { content: [{ type: "text", text: `# Search results for "${args.query}"\n\nFound ${matches.length} skill(s):\n\n${result}` }] };
      }

      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  } catch (error) {
    return { content: [{ type: "text", text: `Error: ${error.message}` }], isError: true };
  }
});

// ─── Start Server ───
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("MRARRANGER Skills MCP Server running on stdio");
}

main().catch(console.error);
