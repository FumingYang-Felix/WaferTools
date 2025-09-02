import csv

input_csv = 'pair_ssim_ss10.csv'
output_csv = 'pair_ssim_ss10_scale07.csv'

with open(input_csv, 'r') as fin, open(output_csv, 'w', newline='') as fout:
    reader = csv.reader(fin)
    writer = csv.writer(fout)
    header = next(reader)
    writer.writerow(header)
    for row in reader:
        try:
            if len(row) < 9:
                continue
            scale = float(row[5])
            if scale >= 0.7:
                writer.writerow(row)
        except Exception:
            continue
print(f'Filtered CSV saved to {output_csv}') 
 
 
 
 
 
 
 
 
 
 
 