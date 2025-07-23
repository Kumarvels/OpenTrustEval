# 🏥 Hospital Daily Operations & Support Chat System – Deep Conversation Example

## Scenario Overview
Hospital staff and patients interact with the support system for scheduling, clinical queries, and operational support. Each response is evaluated for trust, hallucination risk, and HIPAA compliance, with OpenTrustEval Justifications and OTE Notes.

---

## 1. Conversation Flow (20 Steps)

| Step | User Input | System Response | Trust Score | Hallucination Risk | OpenTrustEval Justification | OTE Notes |
|------|------------|-----------------|-------------|--------------------|----------------------------|-----------|
| 1 | Hi, I need to check my appointment. | Please provide your patient ID. | 0.99 | 0.01 | Intent match, no ambiguity. | 👍 |
| 2 | Patient ID: JD123 | Your next appointment is at 2pm with Dr. Smith. | 0.98 | 0.01 | Schedule from EHR. | 👍 |
| 3 | What is Dr. Smith's specialty? | Dr. Smith is a cardiologist. | 0.98 | 0.01 | Staff DB lookup. | 👍 |
| 4 | Can I reschedule? | Yes, available slots: 3pm, 4pm. | 0.97 | 0.01 | Calendar check. | 👍 |
| 5 | Reschedule to 3pm. | Appointment rescheduled to 3pm. | 0.99 | 0.01 | Calendar update confirmed. | 👍 |
| 6 | What are my medications? | You are prescribed Atorvastatin and Metformin. | 0.97 | 0.01 | Medication list from EHR. | 👍 |
| 7 | Can I get a refill? | Yes, refill request sent to pharmacy. | 0.98 | 0.01 | Protocol check, pharmacy notified. | 👍 |
| 8 | What is my last lab result? | Your last cholesterol was 180 mg/dL. | 0.97 | 0.01 | Lab results from EHR. | 👍 |
| 9 | Can I get a copy? | Lab report sent to your email. | 0.98 | 0.01 | Document delivery protocol. | 👍 |
| 10 | Who can access my records? | Only authorized staff per HIPAA. | 0.99 | 0.01 | Policy-based answer. | 👍 |
| 11 | How do I update my contact info? | Update via patient portal or front desk. | 0.98 | 0.01 | Policy and system check. | 👍 |
| 12 | What is the hospital's visiting policy? | Visiting hours: 10am-6pm, 2 visitors max. | 0.97 | 0.01 | Policy from admin DB. | 👍 |
| 13 | Can I request a wheelchair? | Yes, request sent to support staff. | 0.98 | 0.01 | Service request protocol. | 👍 |
| 14 | How do I pay my bill? | Pay online or at billing desk. | 0.98 | 0.01 | Payment options from billing DB. | 👍 |
| 15 | What insurance do you accept? | We accept Aetna, Blue Cross, and Cigna. | 0.97 | 0.01 | Insurance list from admin DB. | 👍 |
| 16 | Can I get a medical certificate? | Yes, request submitted to your doctor. | 0.98 | 0.01 | Protocol check. | 👍 |
| 17 | How do I file a complaint? | Use the portal or contact admin at admin@hospital.com. | 0.98 | 0.01 | Policy and contact info. | 👍 |
| 18 | What is the emergency number? | Call 911 for emergencies. | 0.99 | 0.01 | Policy-based, universal. | 👍 |
| 19 | Can I talk to a nurse? | Connecting you to the nurse station... | 0.99 | 0.01 | Escalation protocol. | 👍 |
| 20 | Thank you! | You're welcome! Stay healthy. | 0.99 | 0.01 | Polite closure. | 👍 |

---

## 2. Mindmap of Conversation Paths

```mermaid
graph TD
    A[Start] --> B[Appointment Inquiry]
    B --> C[Reschedule]
    B --> D[Doctor Info]
    B --> E[Medication]
    E --> F[Refill]
    B --> G[Lab Results]
    G --> H[Get Copy]
    B --> I[Access Policy]
    B --> J[Update Info]
    B --> K[Visiting Policy]
    B --> L[Service Request]
    B --> M[Billing]
    B --> N[Insurance]
    B --> O[Medical Certificate]
    B --> P[Complaint]
    B --> Q[Emergency]
    B --> R[Nurse Escalation]
```

---

## 3. Flow Diagram

```mermaid
flowchart TD
    U[User] --> Q[WebUI]
    Q --> C1[Support/LLM]
    C1 --> C2[Trust & Hallucination Engine]
    C2 --> Q
    Q --> U
    C2 --> S[Admin Dashboard]
```

---

## 4. How Trust & Hallucination Scores Are Computed

- **Trust Score:**
  - Based on data source reliability (EHR, admin DB, LLM generation)
  - Cross-checked with hospital policies and compliance
  - Penalized for ambiguous or unverifiable answers

- **Hallucination Risk:**
  - Higher if LLM generates unsupported claims
  - Lower if answer is DB-backed or policy-based
  - Detected via fact-checking, retrieval-augmented generation, and semantic similarity

- **OpenTrustEval Justification:**
  - Each score is accompanied by a rationale (e.g., “EHR-verified”, “Policy-based”, “LLM-generated with low confidence”)

---

## 5. Best Practices & Lessons Learned

- Always check for HIPAA compliance before displaying patient data.
- Use trust and hallucination scores to flag risky responses.
- Maintain audit logs for all operational queries. 