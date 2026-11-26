"""Tools for building training data for a self-hosted hairline model.

Training pairs are made in reverse: start from a consented photo with a full hairline (the
real "after"), synthesise a receded version (the "before"), and train the model to restore
the real photo. The target is always real hair. No Gemini output is used anywhere here.
"""
