---
name: mrarranger-fb-page-likes
description: "MRARRANGER Facebook Page Likes Ads - ระบบยิงแอดเพิ่ม Follower/Like ของ Facebook Page ครบวงจร รองรับ Campaign Strategy, Audience Targeting, Budget Optimization, Ad Creative Templates, และ Performance Analysis สำหรับ Page Like campaigns โดยเฉพาะ ใช้เมื่อ: (1) ต้องการเพิ่ม follower/like ของ Facebook Page (2) สร้าง Page Like campaign ใน Facebook Ads Manager (3) วาง targeting strategy สำหรับ Page Like ads (4) ตั้งงบประมาณและ optimize ค่าใช้จ่ายต่อ like (5) เขียน ad copy/creative สำหรับ Page Like ads (6) วิเคราะห์ performance ของ Page Like campaign (7) scale campaign ที่ทำกำไร ใช้ skill นี้ทุกครั้งที่มีคำว่า Page Like, เพิ่ม follower, ยิงแอดเพิ่มคนติดตาม, Facebook Like campaign, CPL (cost per like), หรือพูดถึงการเพิ่มจำนวนคนกดถูกใจเพจ Commands: /pagelike /targeting /budget /adcopy /analyze /scale"
---

# MRARRANGER Facebook Page Likes Ads

ระบบยิงแอดเพิ่ม Follower/Like ของ Facebook Page — ครบตั้งแต่ Strategy ไปจนถึง Scale

## Quick Commands

| Command | ใช้งาน | Output |
|---------|--------|--------|
| `/pagelike [page name]` | สร้าง Page Like campaign ครบชุด | Full campaign blueprint |
| `/targeting [niche]` | ออกแบบ Audience targeting | 3-5 audience sets |
| `/budget [งบ/วัน]` | วาง budget strategy | Budget allocation + timeline |
| `/adcopy [page name]` | เขียน Ad copy + creative direction | 3-5 ad variations |
| `/analyze [metrics]` | วิเคราะห์ performance | Insights + action items |
| `/scale [campaign]` | Scale campaign ที่ทำงาน | Scaling playbook |

---

## PART 1: PAGE LIKE CAMPAIGN FUNDAMENTALS

### ทำไมต้องยิง Page Like Ads?

Page Like campaign เหมาะกับ:
- **เพจใหม่** ที่ต้องการ social proof เบื้องต้น (ต่ำกว่า 1,000 likes)
- **เพจ content** ที่ต้องการ base audience สำหรับ organic reach
- **เพจธุรกิจ** ที่ต้องการ credibility ก่อนรัน conversion ads
- **ใช้เป็น funnel step** — ดึงคนมาไลค์เพจ แล้วค่อย retarget ด้วย conversion ads ภายหลัง

### ข้อควรรู้ก่อนเริ่ม

Organic reach ของ Facebook Page ในปี 2025-2026 ต่ำมาก (ค่าเฉลี่ยอยู่ที่ประมาณ 2-5% ของ followers) ดังนั้น Page Likes ไม่ได้แปลว่าคนจะเห็น content ของเราโดยอัตโนมัติ ต้องใช้ร่วมกับ strategy อื่น:
- Organic Reels (อัลกอริทึม Facebook 2026 ให้ reach สูงมากกับ Reels)
- Content ที่สร้าง saves/shares (สัญญาณที่แรงที่สุดสำหรับ algorithm)
- Facebook Groups (engagement rate 80-200%)
- Retargeting ads สำหรับ conversion

---

## PART 2: CAMPAIGN SETUP — STEP BY STEP

### Step 1: Campaign Structure

```
📁 Campaign: [PageName]_PageLikes_[Month][Year]
   Objective: Engagement > Page Likes
   Budget Type: CBO (Campaign Budget Optimization)
   
   📂 Ad Set 1: Interest_[DescriptiveLabel]
      - Audience: Interest-based targeting
      - Budget share: 40%
      
   📂 Ad Set 2: LAL_[DescriptiveLabel]  
      - Audience: Lookalike 1% of page engagers
      - Budget share: 40%
      
   📂 Ad Set 3: Retarget_[DescriptiveLabel]
      - Audience: Video viewers / website visitors (warm)
      - Budget share: 20%
```

### Naming Convention

ใช้ pattern นี้เพื่อให้ filter/analyze ง่าย:
- Campaign: `PageLikes_CBO_[Audience]_[Month][Year]`
- Ad Set: `PageLikes_CBO_[Audience]_[Month][Year] | AdSet_[AudienceDetail]_[Location]_[Age]`
- Ad: `... | Ad_[CreativeType]_[Version]`

ตัวอย่าง: `PageLikes_CBO_Prospecting_Mar2026 | AdSet_Interest_FunFact_TH_18-45 | Ad_Video_V1`

### Step 2: เลือก Objective

ใน Facebook Ads Manager:
1. เลือก Campaign Objective: **Engagement**
2. เลือก Engagement Type: **Page Likes**
3. เลือก Facebook Page ที่ต้องการ

### Step 3: Budget Setting

**สำหรับเพจเริ่มต้น (ทดสอบ):**

| Daily Budget | คาดหวัง Likes/วัน | CPL (THB) โดยประมาณ |
|---|---|---|
| 100-300 THB | 10-40 likes | 3-10 THB |
| 300-700 THB | 40-100 likes | 3-8 THB |
| 700-2,000 THB | 100-300 likes | 3-7 THB |

หมายเหตุ: CPL ในไทยโดยเฉลี่ยอยู่ที่ 2-10 THB ขึ้นอยู่กับ niche, creative quality, และ audience targeting ตัวเลขค่าเฉลี่ย global อยู่ที่ประมาณ $0.35/like

**สูตรคำนวณ Minimum Daily Budget:**
```
(Target Events ÷ 7) × Cost Per Event = Minimum Daily Budget
ตัวอย่าง: (50 likes ÷ 7) × 5 THB = 36 THB/day minimum
```

ต้องการอย่างน้อย 50 events ใน 7 วัน เพื่อออกจาก Learning Phase ของ Meta

---

## PART 3: AUDIENCE TARGETING STRATEGY

### 3-Layer Targeting Framework

**Layer 1: Interest-Based (Cold — ยิงกว้าง)**
- เลือก interests ที่ตรงกับ niche ของเพจ
- ใช้ 3-5 interests ที่ไม่ overlap กันในแต่ละ ad set
- Audience size: 500K - 5M คน (สำหรับไทย)
- Exclude: คนที่ like เพจแล้ว

**Layer 2: Lookalike Audiences (Warm — ยิงแม่น)**
- LAL 1% จาก Page Engagers
- LAL 1% จาก Video Viewers (ถ้ามี video content)
- LAL 1% จาก Website Visitors (ถ้ามี Pixel)
- Audience size: จะถูกกำหนดโดย Meta อัตโนมัติ

**Layer 3: Retargeting (Hot — ยิงซ้ำ)**
- คนที่ดู Video 50%+ ใน 30 วัน
- คนที่ engage กับ post/ad ใน 30 วัน
- Website visitors 30 วัน
- คนกลุ่มนี้มี CPL ต่ำที่สุด เพราะรู้จักเราแล้ว

### Targeting Best Practices

- **Exclude existing followers** ทุกครั้ง — ไม่ต้องจ่ายซ้ำ
- **Exclude Messenger + Instagram placements** — Page Like ads ทำงานดีที่สุดบน Facebook Feed
- **สร้างอย่างน้อย 3 ad sets ที่ไม่ overlap** — เพื่อให้ Meta มี data เปรียบเทียบ
- **Broad targeting + ปล่อย Meta optimize** — ในปี 2026 algorithm ของ Meta ฉลาดมาก ไม่ต้อง narrow เกินไป
- **ใช้ Advantage+ Audience** ถ้ามี data เพียงพอ (500+ conversions)

### Audience Sizing Guide (ตลาดไทย)

| Niche | Estimated Audience | Competition |
|---|---|---|
| ข่าว/Fun Fact/Entertainment | 5-20M | สูง |
| อาหาร/ท่องเที่ยว | 3-15M | สูง |
| เทคโนโลยี/Gadget | 1-5M | ปานกลาง |
| ธุรกิจ/การเงิน | 1-3M | ปานกลาง |
| Niche เฉพาะทาง | 100K-1M | ต่ำ |

---

## PART 4: AD CREATIVE & COPY TEMPLATES

### Creative Principles สำหรับ Page Like Ads

1. **บอก Value Proposition ชัด** — ทำไมต้อง follow เพจนี้?
2. **ใช้ Visual ที่สะดุดตา** — Video > Image > Carousel
3. **Mobile-first เท่านั้น** — 98.5% ของ users ไทยใช้มือถือ
4. **Hook ใน 2 วินาทีแรก** — ถ้าเป็น video ต้อง grab attention ทันที

### Ad Copy Templates

**Template 1: Value-Driven (เน้นประโยชน์)**
```
📌 [Headline]: กด Follow เพื่อ [ประโยชน์ที่ได้]
📝 [Body]:
ถ้าคุณชอบ [topic], เพจนี้สำหรับคุณ!
✅ [ข้อดี 1]
✅ [ข้อดี 2]  
✅ [ข้อดี 3]
กดถูกใจเพจเพื่อไม่พลาดทุก update 👍
```

**Template 2: Curiosity-Driven (เน้นความอยากรู้)**
```
📌 [Headline]: รู้หรือเปล่า? [Interesting Fact]
📝 [Body]:
เพจนี้โพสต์เรื่อง [topic] ที่คุณไม่เคยรู้มาก่อน ทุกวัน
กดถูกใจเพจ แล้วคุณจะไม่เชื่อว่าโลกนี้มีเรื่องแบบนี้ด้วย 🤯
```

**Template 3: Social Proof (เน้นความนิยม)**
```
📌 [Headline]: [จำนวน] คนกำลัง Follow เพจนี้
📝 [Body]:
มากกว่า [จำนวน] คนเลือกติดตามเพจ [ชื่อเพจ] แล้ว
เพราะเราส่ง [content type] ที่ [benefit] ให้ทุกวัน
มา join กัน! กดถูกใจเพจเลย 💙
```

**Template 4: Content Teaser (เน้นตัวอย่าง content)**
```
📌 [Headline]: แบบนี้โพสต์ทุกวัน
📝 [Body]:
[ตัวอย่าง content ที่ดีที่สุดของเพจ]
ชอบแบบนี้? เรามีอีกเยอะ!
กดถูกใจเพจ [ชื่อเพจ] เพื่อรับ content แบบนี้ทุกวัน ✨
```

### Creative Types ที่ทำงานดีที่สุด

| Creative Type | CTR เฉลี่ย | CPL เฉลี่ย | แนะนำสำหรับ |
|---|---|---|---|
| Short Video (< 15 วินาที) | สูงสุด | ต่ำสุด | ทุก niche |
| Reels-style Vertical Video | สูงมาก | ต่ำ | Entertainment, Fun Fact |
| Static Image + Bold Text | ปานกลาง | ปานกลาง | ธุรกิจ, Professional |
| Carousel (แสดงตัวอย่าง content) | ปานกลาง | ปานกลาง | เพจที่มี content หลากหลาย |

---

## PART 5: BUDGET OPTIMIZATION & BIDDING

### Budget Allocation Strategy

**Phase 1: Testing (สัปดาห์ที่ 1-2)**
- Budget: 200-500 THB/day
- สร้าง 3 ad sets, 2-3 ads ต่อ ad set
- ปล่อยให้ Meta optimize
- เป้าหมาย: หา winning audience + creative

**Phase 2: Optimization (สัปดาห์ที่ 3-4)**
- Kill ad sets ที่ CPL สูงกว่า target 2x
- เพิ่ม budget ให้ winning ad sets 20-30% ทุก 3 วัน
- สร้าง ad variations จาก winning creative

**Phase 3: Scaling (เดือนที่ 2+)**
- Budget: ขยายจาก winning campaigns
- เพิ่มไม่เกิน 20% ต่อ 3 วัน (เพื่อไม่ให้ reset learning phase)
- สร้าง LAL audiences จาก new followers

### Bidding Strategy

- **Lowest Cost (แนะนำ)** — ปล่อย Meta หา likes ราคาถูกที่สุด
- ไม่ต้อง set bid cap สำหรับ Page Like campaigns
- ใช้ CBO (Campaign Budget Optimization) เพื่อให้ Meta จัดสรรงบเอง

### CPL Benchmarks (ตลาดไทย)

| Performance | CPL (THB) | Action |
|---|---|---|
| ดีมาก | < 3 THB | Scale ได้เลย |
| ดี | 3-5 THB | Optimize creative แล้ว scale |
| ปานกลาง | 5-8 THB | ทดสอบ audience/creative ใหม่ |
| แย่ | 8-15 THB | Pause แล้ว revamp |
| แย่มาก | > 15 THB | Kill campaign ทันที |

---

## PART 6: PERFORMANCE ANALYSIS & OPTIMIZATION

### KPIs ที่ต้องติดตาม

| Metric | ดูอะไร | Target (ไทย) |
|---|---|---|
| Cost per Like (CPL) | ราคาต่อ like | < 5 THB |
| CTR (Click Through Rate) | % คนคลิก ad | > 1.5% |
| Frequency | คนเห็น ad กี่ครั้ง | < 3 ต่อสัปดาห์ |
| Relevance Score | คุณภาพ ad | > 7/10 |
| Amount Spent | ใช้จ่ายจริง vs budget | ตาม plan |

### Optimization Decision Tree

```
CPL สูงเกินไป?
├── CTR ต่ำ (< 1%)
│   ├── Creative ไม่ดึงดูด → เปลี่ยน creative/video
│   └── Audience ไม่ตรง → ปรับ targeting
├── CTR ดี แต่ CPL สูง
│   ├── Frequency สูง (> 4) → Audience หมด, ขยาย audience
│   └── Competition สูง → เปลี่ยน placement/เวลา
└── CPL ปกติแต่ไม่ได้ likes
    └── Budget ต่ำเกินไป → เพิ่ม budget
```

### Weekly Review Checklist

1. เช็ค CPL ของทุก ad set — kill ตัวที่แพงกว่า average 2x
2. เช็ค Frequency — ถ้าเกิน 3 = audience fatigue, ต้องเปลี่ยน creative หรือขยาย audience
3. เช็ค winning creative — duplicate และทำ variation ใหม่
4. เช็ค audience performance — LAL vs Interest vs Retargeting
5. ปรับ budget allocation ตาม performance

---

## PART 7: SCALING PLAYBOOK

### เมื่อไหร่ถึงจะ Scale?

Scale ได้เมื่อ:
- CPL ต่ำกว่า target อย่างน้อย 3 วันติด
- มี winning ad set อย่างน้อย 2 ตัว
- Frequency ยังต่ำกว่า 2

### Scaling Methods

**Vertical Scaling (เพิ่มงบ ad set เดิม)**
- เพิ่ม 20% ทุก 3 วัน
- ห้ามเพิ่มเกิน 30% ต่อครั้ง — จะ reset learning phase
- Monitor CPL หลังเพิ่ม 24 ชม.

**Horizontal Scaling (เพิ่ม ad sets ใหม่)**
- Duplicate winning ad set + เปลี่ยน audience
- สร้าง LAL 2%, 3% จาก winning audience
- ทดสอบ creative variations ใหม่

**Geographic Scaling (ถ้าเพจไม่จำกัดพื้นที่)**
- เริ่มจากไทย → ขยายไป SEA countries ที่ตรง niche
- ต่างประเทศอาจมี CPL ที่ต่างกันมาก

### Growth Hack: Page Like → Group Invite Funnel

หนึ่งใน use cases ที่ดีที่สุดของ Page Like ads:
1. ยิง Page Like ads → คนมากด like เพจ
2. สร้าง Facebook Group ที่เชื่อมกับเพจ
3. Invite คน like เพจ เข้า Group (ฟรี! เพราะเป็น invitation)
4. Group มี engagement rate สูงกว่าเพจ 10-50x
5. ใช้ Group เป็น community สำหรับขาย product/service

---

## PART 8: ADVANCED STRATEGIES

### Strategy 1: Content-First Approach (แนะนำ)

แทนที่จะยิง Page Like ads ตรงๆ:
1. โพสต์ Reels/Video content ที่ดีมาก (organic)
2. Boost video ที่มี organic engagement ดี → เพิ่ม reach
3. คนที่ดู video 50%+ → Retarget ด้วย Page Like ad
4. คนกลุ่มนี้มี CPL ต่ำมาก เพราะรู้จักเราแล้ว
5. Invite ทุกคนที่ react กับ post ให้มา like เพจ (ฟรี!)

### Strategy 2: Multi-Campaign Funnel

```
🔵 Campaign 1: Video Views (Awareness)
   - Budget: 40% ของ total
   - ยิง video content ดีๆ ให้คนดู
   
🟢 Campaign 2: Page Likes (Consideration)  
   - Budget: 40% ของ total
   - Target: คนที่ดู video จาก Campaign 1
   - CPL จะถูกกว่ายิงตรงมาก
   
🟡 Campaign 3: Conversion (Action)
   - Budget: 20% ของ total
   - Target: Page followers + Group members
   - ขาย product/service
```

### Strategy 3: Invite Button Hack (ฟรี!)

Facebook อนุญาตให้ invite คนที่ react กับ post ของเพจ ให้มา like เพจได้:
1. ไปที่ post ที่มีคน react เยอะ
2. คลิกจำนวน reactions
3. จะเห็นปุ่ม "Invite" ข้างชื่อคนที่ยังไม่ได้ like เพจ
4. คลิก Invite ทีละคน (ทำได้สูงสุดวันละประมาณ 500-1000 คน)
5. Conversion rate จากการ invite อยู่ที่ประมาณ 10-30%

---

## PART 9: REAL-WORLD REFERENCE

### Case Study: Fun Fact Thailand (จาก Ads Manager)

```
Account: Fun Fact Thailand
Account ID: act_957500308379070
Currency: THB | Status: Active

Today's Snapshot:
- Active Campaigns: 7
- Daily Budget: 700 THB/day (100 THB/campaign avg)
- Amount Spent (today): 164.72 THB
- Page Likes (today): 441

Calculated Metrics:
- CPL: 164.72 ÷ 441 = ~0.37 THB/like ← ถูกมากกกก
- ถ้าเฉลี่ย 441 likes/day × 30 days = ~13,230 likes/month
- Monthly spend estimate: 700 × 30 = 21,000 THB
- Monthly CPL estimate: ~1.59 THB/like
```

ข้อสังเกต:
- ใช้ 7 campaigns (หลาย ad sets) = diversified testing
- Budget 700 THB/day = ประมาณ 100 THB ต่อ campaign
- CPL ต่ำมาก → niche "Fun Fact" มี audience กว้าง + competition ต่ำ
- Pattern: ยิงหลาย campaigns พร้อมกัน ให้ Meta เลือก optimize เอง

---

## APPENDIX: QUICK REFERENCE

### Campaign Checklist ก่อนกด Publish

- [ ] Objective: Engagement > Page Likes
- [ ] Exclude existing followers
- [ ] Exclude Messenger + Instagram placements
- [ ] อย่างน้อย 3 ad sets ที่ไม่ overlap audiences
- [ ] อย่างน้อย 2 ad creatives ต่อ ad set
- [ ] Budget ≥ minimum สำหรับออก Learning Phase
- [ ] Naming convention ถูกต้อง
- [ ] Mobile-optimized creative (vertical video preferred)
- [ ] CTA ชัดเจน — บอกให้คน like เพจ

### เวลาที่ดีที่สุดในการยิง (ตลาดไทย)

- ช่วงเช้ามืด 05:00-09:00 — CPL มักจะถูกที่สุด (competition ต่ำ)
- ช่วงเที่ยง 11:00-13:00 — reach สูง
- ช่วงเย็น 17:00-20:00 — engagement สูง โดยเฉพาะ video/Reels
- วันอังคาร-พฤหัสบดี มักมี engagement ดีที่สุด
- แนะนำ: ใช้ automated delivery ปล่อย Meta optimize เอง

### Do's and Don'ts

**Do's:**
- ทดสอบ creative หลายแบบก่อน commit budget
- ใช้ video/Reels เป็นหลัก
- ดึง warm audience มา retarget
- Review performance ทุกสัปดาห์
- Scale ช้าๆ (20% ทุก 3 วัน)

**Don'ts:**
- อย่ายิงกว้างโดยไม่ exclude existing followers
- อย่าใช้ Messenger/Instagram placement สำหรับ Page Like ads
- อย่าเพิ่ม budget เกิน 30% ต่อครั้ง
- อย่าปล่อย campaign ที่ CPL สูง > 2 สัปดาห์
- อย่าซื้อ fake likes — algorithm จะลงโทษ

---

*Skill version: 1.0 | Last updated: March 2026*
*Based on Meta Ads Manager best practices 2025-2026 + real campaign data*
