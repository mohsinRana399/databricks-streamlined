"""
Setup script for the Databricks PDF Upload application.
"""
import os
import sys
import subprocess
from pathlib import Path


def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False


def create_directories():
    """Create necessary directories."""
    print("Creating directories...")
    directories = [
        "logs",
        "src",
        "utils", 
        "notebooks"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")


def setup_environment():
    """Set up environment file."""
    print("Setting up environment configuration...")
    
    if not os.path.exists('.env'):
        if os.path.exists('.env.example'):
            print("Creating .env file from template...")
            with open('.env.example', 'r') as example:
                content = example.read()
            
            with open('.env', 'w') as env_file:
                env_file.write(content)
            
            print("✅ Created .env file from template")
            print("⚠️  Please edit .env file with your actual Databricks credentials")
        else:
            print("❌ .env.example not found")
            return False
    else:
        print("✅ .env file already exists")
    
    return True


def validate_setup():
    """Validate the setup."""
    print("Validating setup...")
    
    required_files = [
        "app.py",
        "config.py", 
        "requirements.txt",
        ".env",
        "src/databricks_client.py",
        "src/databricks_api.py",
        "utils/pdf_processor.py",
        "utils/logger.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    print("✅ All required files present")
    return True


def main():
    """Main setup function."""
    print("🚀 Setting up Databricks PDF Upload Application")
    print("=" * 50)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during package installation")
        return False
    
    # Setup environment
    if not setup_environment():
        print("❌ Setup failed during environment configuration")
        return False
    
    # Validate setup
    if not validate_setup():
        print("❌ Setup validation failed")
        return False
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit the .env file with your Databricks credentials")
    print("2. Run: streamlit run app.py")
    print("3. Open http://localhost:8501 in your browser")
    print("\nFor detailed instructions, see README.md")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
