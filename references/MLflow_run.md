# How the MLflow Run Management Works

The MLflow run management works in three distinct phases:

## Start Phase (start_workflow_run):

Creates a new MLflow run and gets a run_id
Sets initial tags like workflow type and start time
Ends the run (temporarily)
Returns the run_id to be used in subsequent calls


## Continue Phase (continue_workflow_run):

Resumes an existing run using the run_id
Updates tags to track the current step
Returns the run context to be used in a with statement
The run is automatically ended when exiting the with block


## Complete Phase (complete_workflow_run):

Resumes the run one last time
Sets final tags like end time and status
The run is automatically ended when exiting the with block



This three-phase approach allows a single MLflow run to span multiple API calls while maintaining a coherent workflow.