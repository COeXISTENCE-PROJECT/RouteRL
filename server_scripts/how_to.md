1. Clone the project via HTTPS **directly into your directory** and not into any subfolder.
2. In your terminal, run `cd /home/$USER/Milestone-One/server_scripts/` to navigate to this folder. (No need to change anything in this command!)
3. SBatch scripts are already configured. However, you may adjust configurations in `run_simulation.sh` for more resource/shorter wait time.
4. In your terminal, run `sbatch run_simulation.sh` to start running the simulation.
5. Outputs of the SLURM job (`slurm-<job_id>.out`) and the consols prints of the program (`container_printouts/output_<job_id>.txt`) will be in this folder.
6. Just be patient, it is possible that the content of `output_<job_id>.txt` is not updated frequently. Check the status of your job using `squeue --me` in your terminal. Or if you suspect the resources you requested are not actually allocated, you can inspect using `scontrol show jobid -d <job_id>`.
8. Once the simulation ends, all records, plots, .csv files that the simulation normally saves to the disk will be accessible in `Milestone-One` folder as usual.
