# Customer-support agent evaluation starter

This template tests policy-grounded assistance, account privacy, escalation,
incident handling, and whether answers remain clear and empathetic under
pressure.

Before running it:

1. Replace the disabled adapter with your support-agent adapter.
2. Rename `lookup_order`, `lookup_account`, and `check_status` to match emitted
   tool names.
3. Replace all illustrative policy facts with approved current policies.
4. Use synthetic customer and account data only.
5. Review `llm_judge` cases with support and compliance owners before creating
   a baseline.

Never place live credentials, payment data, authentication answers, or private
customer conversations in golden cases or run artifacts.
