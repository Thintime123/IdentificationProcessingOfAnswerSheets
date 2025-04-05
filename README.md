# Intelligent Answer Sheet Recognition and Processing System

## Project Overview
This project is a Python-based intelligent answer sheet recognition and processing system designed to automate the scanning, identification, and scoring of answer sheets. The system can quickly and accurately process multiple answer sheets, generating detailed statistical reports and visualization analysis.

## Key Features
- Batch recognition of answer sheet images
- Automatic scoring and error analysis
- Generation of detailed grade statistics
- Multiple data visualizations (bar charts, pie charts)
- Export results to Excel

## Technical Characteristics
- Image processing using OpenCV
- Statistical chart generation with Matplotlib
- Support for batch processing of multiple answer sheets
- Generation of question error statistics
- Providing intuitive performance analysis

## Dependencies
- OpenCV
- Matplotlib
- Pandas (for Excel export)

## Usage Steps
1. Prepare answer sheet images (stored in `./res/img/` directory)
2. Configure answer key
3. Run the main program
4. View generated statistical reports and charts

## Output Results
- Bar chart: Question error distribution
- Pie chart: Grade distribution
- Excel report: Detailed grade sheet

## Precautions
- Ensure answer sheet images are clear
- Prepare answer key in the specified format
- Image files must be named in sequence (img1.png, img2.png, etc.)

## Future Improvement Directions
- Support for more image formats
- Optimization of recognition algorithms
- Addition of more statistical dimensions