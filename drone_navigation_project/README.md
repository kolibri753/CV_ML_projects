# Task 2 - Classic CV - Drone Navigation

For drone navigation and mapping tasks, choosing the right algorithms for image processing is essential. This project implements a simple baseline approach by converting all images to grayscale and matching them directly **without interpolation or smoothing**.

It leverages classic computer vision techniques including:

**Feature Extraction & Matching (SIFT + FLANN):**
- **SIFT (Scale-Invariant Feature Transform):** extracts robust keypoints that are invariant to scale, rotation, and illumination;
- **FLANN (Fast Library for Approximate Nearest Neighbors):** quickly matches descriptors across images.
  
**Geometric Matching (Homography + RANSAC):**
- computes a homography to align drone-captured images with the global satellite map;
- uses RANSAC (Random Sample Consensus) to filter outliers and ensure robust alignment.

## Output Files

All results are saved to: `/content/drive/My Drive/Colab Notebooks/drone_navigation_project/output/`

- **drone_trajectory.avi** - global map video showing the full drone path, triangle orientation, rotation, and similarity info;
- **drone_trajectory_zoomed.avi** - zoomed-in video showing local crop alignment with contours for each frame;
- **combined_zoom_output_{crop_number}.png** - side-by-side image: original crop + zoomed-in map region with rotation and similarity;
- **route_overview.png** - static map with the full route.

## Environment Setup & Directory Structure

This project is designed to run in **Google Colab**.

### Required Directories on Google Drive

Make sure you have the following paths set up in your Google Drive:

- **Global Map Image:** `/content/drive/My Drive/Colab Notebooks/drone_navigation_project/data/global_map.png`;
- **Drone Crop Images:** `/content/drive/My Drive/Colab Notebooks/drone_navigation_project/data/crops/`;
- **Output Folder:** `/content/drive/My Drive/Colab Notebooks/drone_navigation_project/output/`.

## Dependencies

The required packages are listed in the `requirements.txt` file:

```txt
opencv-contrib-python
numpy
matplotlib

