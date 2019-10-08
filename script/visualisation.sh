
for a in `ls /home/gerald/Documents/These/SystemX/fork/EM_Hyperbolic/RESULTS_SAMPLES/`
    do 
        CUDA_VISBLE_DEVICES=0 python3.7 launcher_tools/visualisation_kmean.py --file /home/gerald/Documents/These/SystemX/fork/EM_Hyperbolic/RESULTS_SAMPLES/$a
done