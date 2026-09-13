# Planwisely Frontend Design Specification

**Version:** 1.0
**Date:** 2026-09-13
**Status:** Active Development

---

## 1. Product Experience Goal

Help salaried professionals (ages 22–40) answer:

> "How much can I safely spend before payday?"

within seconds of opening the app.

---

## 2. Design Principles

| Principle | Description |
|---|---|
| **Calm** | Reduce financial anxiety, not amplify it |
| **Trustworthy** | Financially credible, transparent, honest |
| **Modern** | Clean, current, professional |
| **Premium** | Quality feel without being flashy |
| **Intelligent** | Surface insights, not raw data |
| **Actionable** | Clear next steps, not just information |
| **Accessible** | WCAG AA compliant |
| **Mobile-first** | Touch-friendly, responsive |

---

## 3. Brand Personality

- **Calm:** Reassuring, not alarming
- **Trustworthy:** Transparent, honest about limitations
- **Modern:** Clean, current design language
- **Premium:** Quality without being flashy
- **Intelligent:** Insightful without being robotic
- **Actionable:** Focused on decisions, not just data

---

## 4. Visual Direction

**Desired character:** CALM, TRUSTWORTHY, MODERN, PREMIUM, INTELLIGENT, FINANCIALLY CREDIBLE, ACTIONABLE.

**Avoid:** Flashy crypto look, casino visual language, excessive gradients, excessive glassmorphism, excessive 3D, generic AI SaaS templates, card soup, student-project dashboard aesthetic.

**Use:** Deep navy / blue structure, teal brand identity, restrained supporting blue/cyan, neutral surfaces, generous whitespace, clear typography, strong number readability, restrained shadows, consistent radii, purposeful motion.

---

## 5. Color System

| Token | Usage |
|---|---|
| `--color-primary` | Teal — brand identity |
| `--color-primary-dark` | Deep navy — structure |
| `--color-secondary` | Blue — supporting actions |
| `--color-accent` | Cyan — highlights |
| `--color-surface` | Neutral — backgrounds |
| `--color-surface-raised` | Slightly lighter — cards |
| `--color-text` | Dark — primary text |
| `--color-text-muted` | Gray — secondary text |
| `--color-success` | Green — positive states |
| `--color-warning` | Amber — caution |
| `--color-danger` | Red — errors, overspend |

---

## 6. Typography

- **Font family:** Inter or system-ui fallback
- **Weights:** 400, 500, 600, 700
- **Scale:** 12, 14, 16, 20, 24, 32, 40, 48px
- **Line heights:** 1.2 (headings), 1.5 (body)
- **Number readability:** Tabular numbers for financial figures

---

## 7. Spacing

- **Base unit:** 4px
- **Scale:** 4, 8, 12, 16, 24, 32, 48, 64px
- **Generous whitespace** between sections

---

## 8. Radius

- **Small:** 6px (buttons, inputs)
- **Medium:** 12px (cards)
- **Large:** 16px (modals, elevated surfaces)

---

## 9. Shadows / Elevation

- **Low:** 0 1px 3px rgba(0,0,0,0.1)
- **Medium:** 0 4px 12px rgba(0,0,0,0.08)
- **High:** 0 8px 24px rgba(0,0,0,0.12)

---

## 10. Motion

- **Duration:** 150ms (micro), 250ms (transitions), 400ms (page)
- **Easing:** ease-out (enters), ease-in (exits)
- **Respect prefers-reduced-motion**

---

## 11. Layout

- **Max width:** 1200px (desktop)
- **Grid:** 12-column
- **Gutter:** 24px
- **Mobile breakpoint:** 768px
- **Tablet breakpoint:** 1024px

---

## 12. Information Architecture

1. Safe-to-Spend (primary)
2. Financial position
3. Upcoming obligations
4. Budget status
5. Spending behavior
6. Forecast
7. Recommendations

---

## 13. Navigation

- **Primary:** Top or side navigation
- **Sections:** Dashboard, Transactions, Budget, Forecast, Scenarios, Settings
- **Mobile:** Bottom tab bar

---

## 14. Landing Page

- Hero section with product promise
- Core value propositions
- Social proof (only if verified)
- CTA: Get Started / Sign Up

---

## 15. Authentication

- Supabase Auth integration
- Email/password, OAuth providers
- Forgot password flow
- Session persistence
- Logout clears sensitive caches

---

## 16. Onboarding

- Connect bank / upload CSV
- Set financial goals
- Set payday date
- Explain data usage honestly

---

## 17. Dashboard

Answer within seconds:
- What is my financial position?
- What can I safely spend?
- What obligations are coming?
- Am I on track?
- What needs attention?
- What should I consider doing?

---

## 18. Safe-to-Spend Experience

**Core question:** "How much can I safely spend before payday?"

The interface should explain:
- Value (the number)
- Why (the reasoning)
- Upcoming obligations
- Budget constraints
- Confidence/context

**Status: Design target / backend dependency.** Frontend must not calculate this independently unless backend supports it.

---

## 19. Transactions

Prioritize: Merchant/description, Amount, Date, Category, Search, Filter, Correction where supported.

Avoid exposing excessive ML terminology. Classification confidence only where useful.

---

## 20. Budget

Clearly distinguish:
- Protected/essential obligations (never reduced)
- Flexible spending
- Savings/investment
- Potential adjustment areas

Hard-protected categories must not appear as casual savings targets.

---

## 21. Forecast

Use language: Projected, Estimated, Expected, Based on available data.
Avoid: Guaranteed, Exact, Will happen.

**Do not call the current static/global artifact genuinely personalized.**

---

## 22. Scenarios / Decisions

Prefer: Decision → effect → tradeoff → recommendation.
Rather than exposing raw calculations.

---

## 23. AI Copilot (Future UX)

- Conversational interface
- Calls authorized Planwisely tools
- Transparent about limitations
- Provider-agnostic

## 24. Cards

- One idea per card; consistent radius, restrained shadow
- Financial figures use tabular numbers, large and prominent
- Never bury critical numbers inside dense cards

---

## 25. Tables

- Sticky header; right-aligned numeric columns
- Transactions: merchant/description, amount, date, category; search/filter above the table
- Mobile: collapse tables to card lists; never truncate amounts

---

## 26. Charts

- Purposeful only; label axes; tooltips on hover/touch
- Forecast charts: shaded p10–p90 band labeled Projected/Estimated
- No 3D or gradient excess; accessible palette; provide text summaries

---

## 27. Forms

- Labels always visible (never placeholder-as-label); inline validation
- Currency inputs right-aligned with explicit currency marker
- Preserve input on failed submit; never silent data loss

---

## 28. Modals

- Destructive actions require confirmation
- Escape/backdrop close; focus trap; return focus on close

---

## 29. Alerts / Toasts

- Non-blocking for success/info; errors inline near their source where possible
- Financial warnings state impact plainly (e.g., "Budget exceeded by X")

---

## 30. Loading States

- Skeletons for dashboards and lists
- Never block the Safe-to-Spend figure without explanation

---

## 31. Empty States

- Explain what will appear and how to get it (e.g., "Upload a CSV to see your budget")

---

## 32. Error States

- Human-readable; never render raw backend messages
- Always provide an actionable next step; retry where sensible

---

## 33. Mobile

- First-class priority order: Safe-to-Spend → key status → actions → insights → deeper analytics
- Touch targets ≥ 44px; bottom tab navigation
- Responsive tables; readable charts; accessible forms

---

## 34. Responsive Design

- Breakpoints: 768px (mobile), 1024px (tablet)
- Layouts reflow — they do not merely shrink
- No horizontal scrolling for primary content

---

## 35. Accessibility

- WCAG AA target: adequate contrast, semantic HTML, full keyboard support
- Visible focus states; proper labels; accessible error messages
- Respect `prefers-reduced-motion`; never rely on color alone
- Charts include accessible text alternatives

---

## 36. Financial Trust

- Explain the origin of every number shown
- Hard-protected obligations are never presented as casual savings targets
- Conservative defaults; no dark patterns (no upsells inside budget screens)

---

## 37. Privacy UX

- Plain-language data-use explanations
- Easy access to deletion/account-removal flows once backend support exists
- Collect only what the product genuinely needs

---

## 38. Product Claims

- No fake testimonials, users, customer counts, transaction volumes, savings claims, bank partnerships, security certifications, or awards
- Verified synthetic metrics must always carry their synthetic label

---

## 39. Demo / Synthetic Data

- Every demo dataset is visibly labeled "Demo data"
- Demo data must never silently substitute for failed real-user data

---

## 40. Content Voice

- Calm, plain, specific: "you can safely spend about X before payday"
- No shame-inducing language; no unexplained jargon (p50/p90 only with plain-language explanation)

---

## 41. Frontend Security

- The frontend is NOT an authorization boundary; client `user_id` and localStorage identity are untrusted
- Avoid sensitive persistent browser storage; clear sensitive caches on logout/account change
- Render API/user data as text — no `innerHTML`/`dangerouslySetInnerHTML` unless justified and sanitized
- Do not render raw backend errors; never ship secrets in frontend code

---

## 42. Design-System Architecture

- Design tokens (color, spacing, type, radius, shadow, motion) live in a single source
- Components consume tokens only; no ad-hoc hex values

---

## 43. Page-Level Acceptance Criteria

- Every page ships with: purpose, the primary question it answers, visual hierarchy, loading/empty/error states, mobile behavior, and an accessibility checklist

---

## 44. QA / Launch Gate

Frontend launch requires:

1. Accessibility pass (WCAG AA)
2. Mobile pass on target devices
3. Demo-data labeling verified
4. Product-claim review
5. Safe-rendering (XSS) audit
6. Error-state coverage
