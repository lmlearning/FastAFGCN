#!/bin/bash
# Wrapper script for the memory-lean argument-acceptability checker (solver.py)
# Adapted from generic ICCMA 2019 interface script.

# function for echoing on standard error
function echoerr()
{
    # to remove standard error echoing, please comment the following line
    echo "$@" 1>&2;
}

################################################################
# C O N F I G U R A T I O N
#
# Customize author/version and ensure paths/python command are correct.

# output information
function information()
{
    # Adapted - update if necessary
    echo "FastAFGCN (Python/ONNX version) v0.1" # Placeholder Name
    echo "Lars Malmqvist"    # Placeholder Author
}

# how to invoke the Python solver:
function solver()
{
    local fileinput=$1  # input file with correct path
    local task=$2       # task to solve (e.g., DC-CO, DS-PR)
    local additional=$3 # additional information, i.e., name of the target argument

    # Determine the directory where this script and solver.py reside
    # Using BASH_SOURCE is generally reliable
    local DIR
    DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
    local SOLVER_SCRIPT="$DIR/solver.py" # Assuming solver.py is in the same directory

    if [ ! -f "$SOLVER_SCRIPT" ]; then
        echoerr "ERROR: Solver script not found at $SOLVER_SCRIPT"
        exit 1
    fi

    # Construct the command line for solver.py
    # Using python3 as per the script's shebang
    # Passing --filepath, --task, and --argument as required by solver.py
    echoerr "Executing: python3 $SOLVER_SCRIPT --filepath \"$fileinput\" --task \"$task\" --argument \"$additional\""
    python3 "$SOLVER_SCRIPT" --filepath "$fileinput" --task "$task" --argument "$additional"
}


# --- Supported Formats ---
# The python script reads a specific "p af N" format. Assuming 'apx' is compatible.
formats=""
formats="${formats} apx" # Aspartix format (or compatible)
# formats="${formats} tgf" # Trivial graph format - uncomment if solver.py supports it

# --- Supported Tasks ---
# Based on solver.py logic: Grounded check + ONNX model for decision problems.
# Enumeration (SE/EE) and Dynamic (-D) tasks are NOT supported by solver.py.
tasks=""
tasks="${tasks} DC-CO"   # Decide credulously according to Complete semantics
#tasks="${tasks} DS-CO"   # Decide skeptically according to Complete semantics
#tasks="${tasks} DC-PR"   # Decide credulously according to Preferred semantics
tasks="${tasks} DS-PR"   # Decide skeptically according to Preferred semantics
tasks="${tasks} DC-ST"   # Decide credulously according to Stable semantics
tasks="${tasks} DS-ST"   # Decide skeptically according to Stable semantics
tasks="${tasks} DC-SST"  # Decide credulously according to Semi-stable semantics
tasks="${tasks} DS-SST"  # Decide skeptically according to Semi-stable semantics
#tasks="${tasks} DC-STG"  # Decide credulously according to Stage semantics
#tasks="${tasks} DS-STG"  # Decide skeptically according to Stage semantics
# Grounded is handled directly by solver.py - both DC/DS reduce to checking the single grounded extension
#tasks="${tasks} DC-GR"   # Decide credulously according to Grounded semantics (handled by solve_grounded)
#tasks="${tasks} DS-GR"   # Decide skeptically according to Grounded semantics (handled by solve_grounded)
tasks="${tasks} DC-ID"   # Decide credulously according to Ideal semantics
#tasks="${tasks} DS-ID"   # Decide skeptically according to Ideal semantics


function list_output()
{
    local check_something_printed=false
    printf "["
    if [[ "$1" = "1" ]]; then # List formats
        for format in ${formats}; do
            if [ "$check_something_printed" = true ]; then
                printf ", "
            fi
            printf "%s" "$format"
            check_something_printed=true
        done
    elif [[ "$1" = "2" ]]; then # List problems/tasks
        for task in ${tasks}; do
            if [ "$check_something_printed" = true ]; then
                printf ", "
            fi
            printf "%s" "$task"
            check_something_printed=true
        done
    fi
    printf "]\n"
}

# --- Main Execution Logic ---
function main()
{
    if [ "$#" = "0" ]; then
        information
        exit 0
    fi

    local local_problem=""
    local local_fileinput=""
    local local_format="apx" # Default format assumption, Python script doesn't seem to use it
    local local_additional=""
    # local_filemod removed as dynamic solver is not supported

    local local_task_valid=""

    while [ "$1" != "" ]; do
        case $1 in
            "--formats")
                list_output 1
                exit 0
                ;;
            "--problems")
                list_output 2
                exit 0
                ;;
            "-p")
                shift
                local_problem=$1
                ;;
            "-f")
                shift
                local_fileinput=$1
                ;;
            "-fo")
                shift
                local_format=$1 # Format argument is parsed but not used by solver.py
                ;;
            "-a")
                shift
                local_additional=$1
                ;;
            # "-m" case removed
            *)
                echoerr "Unknown option: $1"
                # exit 1 # Optional: exit on unknown options
                ;;
        esac
        shift
    done

    # Validate task
    if [ -z "$local_problem" ]; then
        echoerr "ERROR: Task (-p) missing."
        exit 1
    else
        for local_task in ${tasks}; do
            if [ "$local_task" = "$local_problem" ]; then
                local_task_valid="true"
                break # Found a valid task
            fi
        done
        if [ -z "$local_task_valid" ]; then
            echoerr "ERROR: Task '$local_problem' is not supported by this solver."
            echoerr "Supported tasks are: $(list_output 2 | tr -d '\n')"
            exit 1
        fi
    fi

    # Validate input file
    if [ -z "$local_fileinput" ]; then
        echoerr "ERROR: Input file (-f) missing."
        exit 1
    fi
    if [ ! -f "$local_fileinput" ]; then
         echoerr "ERROR: Input file '$local_fileinput' not found."
         exit 1
    fi

     # Validate additional argument for relevant tasks (DC/DS)
    if [[ "$local_problem" == "DC-"* || "$local_problem" == "DS-"* ]]; then
        if [ -z "$local_additional" ]; then
            echoerr "ERROR: Argument (-a) required for task '$local_problem'."
            exit 1
        fi
    fi

    # Call the solver function
    # The format ($local_format) is not passed as solver.py doesn't use it
    local res
    res=$(solver "$local_fileinput" "$local_problem" "$local_additional")

    # Output the result directly (solver.py outputs YES/NO)
    echo "$res"

    # No parsing needed as solver.py output matches ICCMA format for DC/DS tasks
    # parse_output $local_problem "$res"
}

# Execute main function with all script arguments
main "$@"
exit 0