"""Portfolio Intelligence (V2).

Gives the AI layer the context it needs to answer "should this be added to *my*
portfolio?" rather than "is this a good stock?". Portfolio state, allocation,
sizing, and risk live here; the AI reaches them through a section renderer
registered on the existing prompt-section registry, so the prompt builder and
orchestration flow are untouched.
"""
