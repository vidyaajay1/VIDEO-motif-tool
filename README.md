# Visual Integration of Drosophila Enhancer Organization (VIDEO)

## Access the web-app here:  [VIDEO-motif-tool](http://ec2-3-150-143-155.us-east-2.compute.amazonaws.com)  

VIDEO is a user-friendly tool for visualizing Drosophila transcription factor binding motifs in promoter proximal regions of genes. More details, applications 
and case studies using VIDEO can be found in the preprint: 

Ajay, V., Laughner, N., Cahan, P., and Andrew, D. J. (2025). VIDEO - Visual Integration of Drosophila Enhancer
Organization: A tool for integrating and visualizing chromatin accessibility, in vivo transcription factor binding and motif
occurrence in tissue-specific differentially expressed genes. In bioRxiv. https://doi.org/10.1101/2025.10.27.684897

VIDEO was implemented using FastAPI (v0.95.2) backend and a React.js (v18.2.0) frontend. 
Core dependencies include FIMO (v4.12.0), pyfaidx (v0.5.9), Pybedtools (v0.12.0), and Plotly (v6.2.0). 
The application was deployed on an EC2 Instance with Redis as broker. 
Reference genome FASTA and GTF annotation (dm6, release 6.45) were downloaded from FlyBase;
motif libraries were bulk downloaded from JASPAR (2024 update).
