# Topic: Lightweight Neural Architectures for Table Cell Detection (LTCD)

This project explores new approaches to table detection in document images and aims to develop efficient models for real time inference. To achieve the goal next work was done:
1. create a dataset. A synthetic dataset generator was written.
2. design models architecture and approaches.
3. train models.
4. evaluate models.

During the exploration phase three distinct approaches were implemented. 
1. Keypoint detector.
2. Heatmap segmentation: predicts all keypoints on a single heatmap.
3. Autorgressive predictor: predicts normalized points.

As a result, a unique architectures were proposed with one more in development phase
