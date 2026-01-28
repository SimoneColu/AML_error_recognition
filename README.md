# Mistake Detection in Procedural Activities

## Step Localization Approach

For the extension task (Task Verification), we initially attempted to use ActionFormer for automatic step localization. We tested with different feature dimensions:
- [inferenza_finale_256.csv](inferenza_finale_256.csv) - 256-dimensional features
- [inferenza_finale_768.csv](inferenza_finale_768.csv) - 768-dimensional features

However, ActionFormer predictions turned out to be quite noisy and performed poorly on our dataset. The predicted step boundaries were unreliable, which would negatively impact downstream task verification performance.

As a result, for **SUBSTEP 1** (Recipe step localization), we decided to use [step_annotations.json](step_annotations.json) as an oracle. This provides ground truth step boundaries and serves as a stronger baseline, allowing us to isolate and evaluate the performance of the task verification components without the noise introduced by imperfect step localization.

This approach lets us focus on the core task verification challenge while establishing an upper bound on what's achievable with perfect step segmentation.
