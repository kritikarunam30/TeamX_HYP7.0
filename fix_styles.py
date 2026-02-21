#!/usr/bin/env python3
"""Fix BodyText style references in report_generator.py"""

# Read the file
with open('report_generator.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace all occurrences
content = content.replace("self.styles['BodyText']", "self.styles['MedicalBodyText']")

# Write back
with open('report_generator.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Successfully replaced all BodyText occurrences with MedicalBodyText")
print(f"Total replacements made in report_generator.py")
