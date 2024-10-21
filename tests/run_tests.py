import subprocess
import os

def run_tests():
    # Define the path to the tests directory
    tests_dir = os.path.dirname(os.path.abspath(__file__))  # Current directory of this script
    project_root = os.path.dirname(tests_dir)  # One level up from tests directory
    
    # Command to run pytest with the specified arguments
    command = ["pytest", tests_dir, "-p", "no:warnings", "--pyargs"]
    
    # Print the command being executed
    print("Running command:", " ".join(command))
    
    # Execute the command
    result = subprocess.run(command, cwd=project_root)
    
    # Check for the exit code to determine success
    if result.returncode == 0:
        print("All tests passed successfully.")
    else:
        print("Some tests failed. Check the output above for details.")
        exit(result.returncode)

if __name__ == "__main__":
    run_tests()
