<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,15,20,24,30&height=120&section=header&text=Srinivas%20Dharpally&fontSize=42&fontColor=ffffff&animation=twinkling"/>

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&duration=3000&pause=1000&color=58A6FF&center=true&vCenter=true&width=700&lines=ML+Engineer+%C2%B7+Full+Stack+Developer+%C2%B7+Geospatial+AI;Production+CV+pipelines+on+drone+%26+satellite+imagery;Django+%C2%B7+React+%C2%B7+PostGIS+%C2%B7+PyTorch+%C2%B7+LangGraph;From+dataset+%E2%86%92+model+%E2%86%92+API+%E2%86%92+map+visualization)](https://git.io/typing-svg)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/srinivas-dharpally/)
[![Portfolio](https://img.shields.io/badge/Portfolio-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://portfolio-cnu1328.netlify.app/)
[![Email](https://img.shields.io/badge/Email-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:srinivas.dharpally@vassardigital.ai)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/cnu1328)

</div>

---

## About Me

**R&D Engineer @ [VassarDigital.ai](https://vassardigital.ai)** · CSE @ RGUKT Basar (CGPA 9.18)

I build full-stack geospatial platforms end-to-end — backend APIs, spatial analysis engines, ML model deployment, and production DevOps. I own ~80% of a capital-region spatial decision-support platform and have shipped **4 production drone-ML pipelines** plus state-scale satellite models.

Passionate about moving deeper into **AI/ML** while leveraging strong full-stack and GIS foundations.

> *Select ML project write-ups will be published as documentation repos when policy allows.*

---

## ML & AI Work

### Drone Imagery Computer Vision (4 Production Pipelines)

End-to-end ownership: dataset creation → model R&D → spatial business logic → prediction APIs → portal integration.

| Pipeline | Highlights |
|----------|------------|
| **Unauthorized Cultivation** | Dual-model approach evolved to **encoder + dual-decoder** architecture (detection + boundary); spatial clipping for authorized vs unauthorized zones |
| **Illegal Construction** | Building detection on drone imagery; spatial clipping for compliance analysis |
| **Green Cover Monitoring** | Vegetation detection; biodiversity hotspot and green-cover classification |
| **Road Network** | Single-class road detection with constructed vs planned KM extraction; multi-class model in progress |

### Satellite Remote Sensing (State-Scale)

- **Aqua Bodies Detection** — Planet imagery band R&D, model training, **~88% accuracy**, statewide inference
- **Built-up Area Detection** — spectral band analysis, model iteration, **~90% accuracy**, statewide predictions

### Vegetation Segmentation (SAM3 Pipeline)

- **SAM3** text-prompted semantic segmentation on high-resolution orthomosaics
- **CHM-based height classification** (grass, bush, small tree, tall tree) from DEM canopy height model
- GIS polygon export for production map layers
- *Documentation repo coming soon*

### NLP & Classification

- **INDIC News Scraper** — IIIT Hyderabad internship; site-specific scraping across Indian news sources
- Aggregated **50+ CSV datasets**, feature engineering, HTML content classification model

### AI Applications

- **ShapeCoach.ai** — LangGraph multi-agent fitness system, RAG over 900+ exercises, personalized workout/diet plans, OpenAI chatbot *(see Freelance section)*

---

## Full Stack & Platform Work

### Geospatial Decision Platform (~80% Backend Ownership)

- Land allotment, monetization, theme-city zoning, land acquisition tracking, infra-zone layouts, LPS summary, estate management
- **Cassandra → PostgreSQL+PostGIS** migration — **60% geospatial query improvement**, unified data source
- **Config-driven View Creation Module** — upload shapefiles, N-level aggregations, dynamic dashboards, GeoServer layer publishing; reduced new-view backend effort to **near zero**
- Config-driven GIS pipeline: shapefile processing **5–6 hours → ~10 minutes** (~95% improvement)
- **RBAC** via Keycloak for secure department portal access
- Streamlit-based GIS utility tool for internal operations

### Public & Citizen-Facing Portals

- **Farmer/LPS User Portal** — SMS OTP authentication, plot search via Elasticsearch, map visualization, PII-aware farmer name masking
- **Housing Module** — API integration with client-provided external data sources

### Metadata & Data Pipeline

- Shapefile upload APIs with **revert-to-previous-version** support
- Table data processing, ingestion, and Elasticsearch indexing
- Full Elasticsearch R&D — install, column-level insert APIs, search fetch APIs
- Elasticsearch setup and deployment for additional government projects (PRRD)

---

## Geospatial Analysis Tools

| Tool | What I Built |
|------|--------------|
| **Query Analysis** | Filter and run analysis across **520+ map layers**; export results as CSV, PDF, KML, KMZ, SHP, DWG |
| **Buffer & Overlay Analysis** | Point/polygon/line/mosaic buffer analysis with spatial clipping; user-scoped results (6 APIs) |
| **Nearest Neighbour Analysis** | Utility distance from polygon/point; **pgRouting + Dijkstra** shortest-path on trunk/LPS roads; Rank 1–5 distance classification; straight-line fallback when roads unavailable |
| **Drawing Tool** | Polygon select + multi-layer intersect download in **7 export formats** (CSV, PDF, KML, KMZ, SHP, DWX, DWG) |

---

## DevOps, Reliability & Security

- **Night Watcher (Monit)** — self-healing monitoring with real-time alerts for Backend, UI, Keycloak, User Management
- **Jenkins CI/CD** — automated deployments, **~80% reduction** in deployment effort
- **Security Audits** — Nginx hardening, User Management remediation, backend audit fixes, GeoServer PII encryption
- **Disaster Management** — API authorization as part of platform security audit

---

## Freelance & Side Projects

### Amplify Insights — Full Stack Developer

| Project | Stack | Impact |
|---------|-------|--------|
| **Media Scan** | Python, Django | News aggregation, multilingual summaries (Te/Eng), YouTube analytics for 50+ channels — **4 hrs → 1 min** report time |
| **Voter Analytics** | Python, Django, PostgreSQL | PDF extraction, AI-assisted regex, survey workflow with RBAC, district/constituency insights |
| **Action Management** | Python, Django | Real-time action tracking, audit trails, RBAC, notifications |
| **KCR Cup** | Python, Django | Cricket tournament platform — live scoring, stats, leaderboards |

### ShapeCoach.ai — AI + Full Stack Engineer

- Cross-platform fitness app (Android, iOS, Web) — **React Native (Expo) + Django**
- **LangGraph** multi-agent system with **RAG** over 900+ exercise dataset
- Personalized 1-month workout + 7-day diet plan generation
- OpenAI Agent SDK chatbot for real-time fitness assistance
- Razorpay payment gateway with signature verification and subscriptions

### Open Source

| Project | Description |
|---------|-------------|
| [**RGUKT E-MART**](https://github.com/cnu1328/emart-react) | React/Node marketplace — Google OAuth, JWT, cart, wishlist, admin |
| [**Portfolio**](https://github.com/cnu1328/portfolio) | Personal developer portfolio site |
| [**Airline Reservation**](https://github.com/cnu1328/airline-reservation-system) | Java Swing desktop app with JDBC |
| [**Calendar App**](https://github.com/cnu1328/React-Calendar) | React calendar with events and todos |

---

## Tech Stack

**ML & AI**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=flat-square&logo=openai&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=flat-square&logo=langchain&logoColor=white)
![RAG](https://img.shields.io/badge/RAG-FF6F00?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)

**Full Stack**

![Django](https://img.shields.io/badge/Django-092E20?style=flat-square&logo=django&logoColor=white)
![React](https://img.shields.io/badge/React-61DAFB?style=flat-square&logo=react&logoColor=black)
![Next.js](https://img.shields.io/badge/Next.js-000000?style=flat-square&logo=nextdotjs&logoColor=white)
![Node.js](https://img.shields.io/badge/Node.js-339933?style=flat-square&logo=nodedotjs&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=flat-square&logo=typescript&logoColor=white)
![Java](https://img.shields.io/badge/Java-ED8B00?style=flat-square&logo=openjdk&logoColor=white)

**Geospatial**

![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=flat-square&logo=postgresql&logoColor=white)
![PostGIS](https://img.shields.io/badge/PostGIS-336791?style=flat-square&logo=postgresql&logoColor=white)
![GeoServer](https://img.shields.io/badge/GeoServer-78BE20?style=flat-square)
![GeoPandas](https://img.shields.io/badge/GeoPandas-306998?style=flat-square&logo=python&logoColor=white)

**DevOps & Tools**

![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![Jenkins](https://img.shields.io/badge/Jenkins-D24939?style=flat-square&logo=jenkins&logoColor=white)
![Nginx](https://img.shields.io/badge/Nginx-009639?style=flat-square&logo=nginx&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=flat-square&logo=linux&logoColor=black)
![Elasticsearch](https://img.shields.io/badge/Elasticsearch-005571?style=flat-square&logo=elasticsearch&logoColor=white)

---

## GitHub Stats

<div align="center">

<img height="165" src="https://github-readme-stats.vercel.app/api?username=cnu1328&show_icons=true&theme=tokyonight&hide_border=true&include_all_commits=true&count_private=true" alt="GitHub Stats"/>
<img height="165" src="https://github-readme-streak-stats.herokuapp.com/?user=cnu1328&theme=tokyonight&hide_border=true" alt="GitHub Streak"/>

<img src="https://github-readme-stats.vercel.app/api/top-langs/?username=cnu1328&layout=compact&theme=tokyonight&hide_border=true&langs_count=8" alt="Top Languages"/>

<img src="https://github-readme-activity-graph.vercel.app/graph?username=cnu1328&theme=tokyo-night&hide_border=true&area=true" alt="Activity Graph"/>

</div>

---

## Contribution Snake

*Updates daily via GitHub Actions — the snake eats my contribution graph as it moves across the grid.*

<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/cnu1328/cnu1328/output/github-snake-dark.svg" />
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/cnu1328/cnu1328/output/github-snake.svg" />
  <img alt="GitHub contribution snake animation" src="https://raw.githubusercontent.com/cnu1328/cnu1328/output/github-snake.svg" />
</picture>

</div>

---

<div align="center">

**Thanks for visiting!** Feel free to reach out for collaborations on ML, geospatial AI, or full-stack projects.

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,15,20,24,30&height=80&section=footer"/>

</div>
