import streamlit as st

def show():
    st.header("Retention Strategy Overview")

    st.subheader("Objective")
    st.markdown("""
    Reduce churn by improving early client experience, supporting premium plan users, repairing trust, and enhancing service reliability. All recommendations are based on observed churn patterns and are designed to be fair, actionable, and compliant.
    """)

    st.subheader("Strategic Focus Areas")

    # Grouped Section: New Client Engagement
    st.markdown("### A. New Client Engagement (Tenure < 24 Months)")
    st.markdown("""
    These actions target clients in their first two years, when churn risk is highest.
    """)

    st.markdown("**1. Structured Onboarding**")
    st.markdown("""
    - Week 1: Welcome message  
    - Month 3 and 9: Satisfaction surveys  
    - Month 12: Loyalty gesture (e.g., temporary upgrade)  
    - Eligibility based on tenure and service tier
    """)

    st.markdown("**2. Charge-Based Outreach**")
    st.markdown("""
    Clients who have spent over **2,600** early in their contract tend to be at higher risk of leaving.

    **Action Plan:**
    - **Spot early signs**: Flag new clients in this spending range during their first few months  
    - **Reach out early**: Check in around Month 1 and Month 3 to offer support and reinforce value  
    - **Tailor onboarding**:
        - **Under 300**: Standard onboarding (low risk)  
        - **300–2,600**: Light follow-up  
        - **2,600–5,400**: Extra attention  
        - **Over 5,400**: Reinforce expectations and satisfaction  

    **Plan-Fit Counseling:**
    - Offer a **voluntary plan review** for high spenders (e.g., those that are provisioned to arrive into the high spending bracket before the end of 24 months)  
    - Ensure clients understand what their plan includes and whether a **lower-cost alternative** could meet their needs  
    - Frame as a **trust-building service**, not a downgrade push  
    - Document all recommendations and ensure **no pressure or bias** in guidance  
    - Track impact on satisfaction and retention

    **Safeguards:**
    - Eligibility based on objective spend thresholds and tenure  
    - No demographic or behavioral profiling  
    - Messaging must be neutral, informative, and opt-in
    """)

    st.markdown("**3. Event-Based Promotions**")
    st.markdown("""
    Timely perks can reinforce value perception and reduce early churn.

    **Eligibility:**
    - Tenure < 12 months  
    - Plan: Premium or Fiber  
    - Trigger: Seasonal campaigns, anniversaries, or satisfaction milestones

    **Actions:**
    - Speed boost, streaming add-on, or loyalty bonus after survey  
    - Opt-in only, clearly communicated  
    - Track retention impact

    **Safeguards:**
    - No demographic targeting  
    - Eligibility based on tenure, plan, and engagement history  
    - Consistent rule application across regions
    """)

    # Remaining Strategic Areas
    st.markdown("### B. Additional Retention Strategies")

    st.markdown("**4. Trust Repair via Feedback Signals**")
    st.markdown("""
    Feedback containing terms like “emails” and “promised” often flags unresolved expectations.

    - Add transparency checklist to onboarding  
    - Clarify service limits with a “What to Expect” message  
    - Investigate unresolved support tickets and flagged emails  
    - Prioritize resolution and monitor churn in affected accounts
    """)

    st.markdown("**5. Silent Segment Engagement**")
    st.markdown("""
    Clients with no feedback or support contact show limited visibility.

    - Identify silent accounts  
    - Send check-in messages  
    - Encourage survey participation with small non-monetary incentives  
    - Monitor churn in this segment
    """)

    st.markdown("**6. Evening Service Reliability**")
    st.markdown("""
    Performance issues are more frequent during evening peak hours.

    - Audit service quality by region and time slot  
    - Communicate with affected clients  
    - Offer opt-in updates and goodwill gestures based on service metrics
    """)

    st.markdown("**7. Regional and Household Insights (Applied Responsibly)**")
    st.markdown("""
    Retention patterns vary modestly by region and household type.

    **Observed Patterns:**
    - Slightly higher retention among female clients, couples, and families  
    - Region North shows modestly better performance

    **Actions:**
    - Prioritize upgrades in high-churn areas  
    - Ensure onboarding and messaging are inclusive  
    - Promote family-friendly features without targeting offers
    """)

    st.subheader("Monitoring and Evaluation")
    st.markdown("""
    Strategy performance should be reviewed quarterly.

    **Key Metrics:**
    - Churn by tenure group  
    - Retention following onboarding or perks  
    - Feedback response rate  
    - Churn among silent and trust-repair segments

    **Additional Steps:**
    - Tag open-ended feedback by theme  
    - Route themes to relevant teams for resolution tracking  
    - Pilot interventions before scaling
    """)

    st.subheader("Compliance Safeguards")
    st.markdown("""
    All actions must comply with non-discrimination regulations:

    - Use objective criteria (e.g., tenure, service tier)  
    - Avoid targeting based on demographics or predicted behavior  
    - Document and apply eligibility rules consistently
    """)
