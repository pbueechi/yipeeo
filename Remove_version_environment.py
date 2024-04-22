def remove_versions_from_requirements(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Remove version numbers from each line
    clean_lines = [line.split('==')[0] for line in lines]

    with open(output_file, 'w') as f:
        f.write('\n'.join(clean_lines))

# Example usage
input_file = '/home/nluintel/shares/home/Codes/yipeeo/spain/yipeeo/requirementsorig.txt'
output_file = '/home/nluintel/shares/home/Codes/yipeeo/spain/yipeeo/requirements_nirajan.txt'
remove_versions_from_requirements(input_file, output_file)

# Command line implementation is more elegant
# sed 's/==.*//' requirements.txt > requirements_no_versions.txt

