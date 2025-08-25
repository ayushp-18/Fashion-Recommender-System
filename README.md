# Fashion Recommender System

## Brief One Line Summary
A content-based fashion recommender system using ResNet50 for feature extraction and k-NN for similarity matching.

## Overview
Recommender systems are widely used in e-commerce to improve customer engagement by suggesting relevant products. In this project, we build a fashion recommender system that uses computer vision techniques to recommend visually similar fashion items. The system extracts image features using ResNet50 and finds similar items using k-Nearest Neighbors.

## Problem Statement
- Provide real-time recommendations of visually similar fashion products.  
- Use computer vision for feature extraction instead of manual tagging.    

## Dataset
- Fashion product images (custom/local dataset with ~5,000 images).  
- Preprocessed and embedded into feature vectors for similarity matching.  

## Tools and Technologies
- Python  
- TensorFlow / Keras  
- Scikit-learn  
- NumPy, Pandas  
- OpenCV, PIL  
 

## Methods
- Image preprocessing and feature extraction using **ResNet50** (pre-trained on ImageNet).  
- Feature normalization and storage as embeddings.  
- Similarity search using **k-Nearest Neighbors (Euclidean distance)**.  
  

## Key Insights
- Deep learning features (ResNet50) capture visual similarity effectively.  
- k-NN allows efficient similarity search in embedding space.  
- Real-time recommendation enhances user experience in fashion retail applications.  


## Results & Conclusion

Successfully built a recommender system that suggests visually similar fashion items.

Achieved real-time recommendations using ResNet50 embeddings and k-NN search.

Conclusion: The system demonstrates how deep learning and computer vision can be applied in e-commerce to personalize product discovery and improve customer engagement.

## Future Work

Scale system with a larger product catalog.

Integrate with collaborative filtering for hybrid recommendations.


