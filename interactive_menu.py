"""
Interactive Menu System for Stock Prediction Pipeline

This script provides an easy-to-use menu interface for running different
components of the stock prediction system, including the new production
model runner.
"""

import os
import sys
import subprocess
from datetime import datetime

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_banner():
    """Print the application banner."""
    print("=" * 70)
    print("             INTEGRATED STOCK PREDICTION SYSTEM")
    print("                     Production Menu v2.0")
    print("=" * 70)
    print()

def print_menu():
    """Print the main menu options."""
    print("Available Operations:")
    print()
    print("1. Data Collection Only")
    print("2. Model Training")
    print("3. Model Evaluation")
    print("4. Full Pipeline (Data + Train + Evaluate)")
    print("5. Generate Reports")
    print("6. Hyperparameter Optimization")
    print("7. >> LIVE MODEL PREDICTIONS <<  [NEW]")
    print("8. Model Explainability Analysis")
    print("9. System Status & Diagnostics")
    print("0. Exit")
    print()
    print("-" * 70)

def run_command(command, description):
    """Run a system command with proper error handling."""
    print(f"\n{description}")
    print("=" * len(description))
    print(f"Executing: {command}")
    print("-" * 50)
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=False, text=True)
        print("\n" + "=" * 50)
        print("✓ Operation completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Operation failed with error code: {e.returncode}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {str(e)}")
        return False

def check_system_status():
    """Check system status and requirements."""
    print("\nSYSTEM STATUS & DIAGNOSTICS")
    print("=" * 40)
    
    # Check Python version
    python_version = sys.version.split()[0]
    print(f"Python Version: {python_version}")
    
    # Check if required directories exist
    required_dirs = ['src', 'data', 'configs', 'models', 'logs']
    for dir_name in required_dirs:
        status = "✓" if os.path.exists(dir_name) else "✗"
        print(f"Directory {dir_name}: {status}")
    
    # Check for config file
    config_status = "✓" if os.path.exists('configs/config.yaml') else "✗"
    print(f"Configuration file: {config_status}")
    
    # Check for trained models
    if os.path.exists('models'):
        model_files = [f for f in os.listdir('models') if f.endswith('.pth')]
        print(f"Trained models available: {len(model_files)}")
        if model_files:
            latest_model = max(model_files, key=lambda x: os.path.getmtime(os.path.join('models', x)))
            mod_time = datetime.fromtimestamp(os.path.getmtime(os.path.join('models', latest_model)))
            print(f"Latest model: {latest_model} ({mod_time.strftime('%Y-%m-%d %H:%M')})")
    else:
        print("Trained models available: 0")
    
    print()

def run_live_predictions():
    """Run the live prediction pipeline."""
    clear_screen()
    print_banner()
    print("LIVE MODEL PREDICTIONS")
    print("=" * 30)
    print()
    print("This will:")
    print("• Load the latest trained hybrid model")
    print("• Fetch current market data for configured symbols")
    print("• Generate real-time predictions")
    print("• Save results with confidence scores")
    print()
    
    confirm = input("Continue with live prediction? (y/N): ").strip().lower()
    if confirm in ['y', 'yes']:
        success = run_command(
            "python scripts\\model_runner.py",
            "Running Live Prediction Pipeline"
        )
        
        if success:
            print("\n📊 Prediction results have been saved to results/live_predictions/")
            print("📝 Detailed logs available in logs/ directory")
        
        input("\nPress Enter to return to main menu...")
    else:
        print("Operation cancelled.")

def main_menu():
    """Main menu loop."""
    while True:
        clear_screen()
        print_banner()
        print_menu()
        
        try:
            choice = input("Select an option (0-9): ").strip()
            
            if choice == '0':
                print("\nThank you for using the Stock Prediction System!")
                break
                
            elif choice == '1':
                success = run_command(
                    "python main.py --mode collect",
                    "Data Collection Pipeline"
                )
                input("\nPress Enter to continue...")
                
            elif choice == '2':
                success = run_command(
                    "python main.py --mode train",
                    "Model Training Pipeline"
                )
                input("\nPress Enter to continue...")
                
            elif choice == '3':
                model_path = input("\nEnter model path (or press Enter for latest): ").strip()
                cmd = "python main.py --mode evaluate"
                if model_path:
                    cmd += f" --model-path {model_path}"
                
                success = run_command(cmd, "Model Evaluation Pipeline")
                input("\nPress Enter to continue...")
                
            elif choice == '4':
                print("\nFull Pipeline Options:")
                print("1. Standard training")
                print("2. With hyperparameter optimization")
                
                sub_choice = input("Select option (1-2): ").strip()
                
                if sub_choice == '1':
                    cmd = "python main.py --mode full"
                elif sub_choice == '2':
                    cmd = "python main.py --mode full --optimize"
                else:
                    print("Invalid option!")
                    input("Press Enter to continue...")
                    continue
                
                success = run_command(cmd, "Full Training Pipeline")
                input("\nPress Enter to continue...")
                
            elif choice == '5':
                success = run_command(
                    "python main.py --mode report",
                    "Report Generation"
                )
                input("\nPress Enter to continue...")
                
            elif choice == '6':
                success = run_command(
                    "python main.py --mode train --optimize",
                    "Hyperparameter Optimization"
                )
                input("\nPress Enter to continue...")
                
            elif choice == '7':
                run_live_predictions()
                
            elif choice == '8':
                model_path = input("\nEnter model path (or press Enter for latest): ").strip()
                cmd = "python main.py --mode explain"
                if model_path:
                    cmd += f" --model-path {model_path}"
                
                success = run_command(cmd, "Model Explainability Analysis")
                input("\nPress Enter to continue...")
                
            elif choice == '9':
                check_system_status()
                input("\nPress Enter to continue...")
                
            else:
                print("\n✗ Invalid option! Please select 0-9.")
                input("Press Enter to continue...")
                
        except KeyboardInterrupt:
            print("\n\nOperation interrupted by user.")
            break
        except Exception as e:
            print(f"\n✗ Unexpected error: {str(e)}")
            input("Press Enter to continue...")

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        sys.exit(1)