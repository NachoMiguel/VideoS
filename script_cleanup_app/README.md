# Script Cleanup Application

Standalone application to clean and optimize scripts for TTS processing.

## Usage

1. **Place your script** in the `input/` folder
2. **Run the cleanup**:
   ```bash
   python main.py input/your_script.txt
   ```
3. **Get cleaned script** in the `output/` folder

## Examples

```bash
# Clean a script from input folder
python main.py input/my_script.txt

# Clean a script from anywhere
python main.py ../script_tests/generated_script_20250715_140819_okksWNdNaTk.txt

# Clean a script from desktop
python main.py ~/Desktop/new_script.txt
```

## What It Does

1. **Text Cleaner** - Fixes encoding, replaces voice-unfriendly characters
2. **Name Corrections** - Fixes ASR mistakes (Vanam → Jean-Claude Van Damme)
3. **Entity Variations** - Prevents repetitive full names (Jean-Claude Van Damme → Jean-Claude)
4. **TTS Optimization** - Converts abbreviations, numbers to words
5. **Final Validation** - Ensures script is ready for TTS

## Output

- **Cleaned script** saved to `output/` with timestamp
- **Before/after comparison** shown in terminal
- **Processing statistics** and improvements listed

## Requirements

- Python 3.8+
- Access to backend services (text_cleaner, entity_variation_manager) 