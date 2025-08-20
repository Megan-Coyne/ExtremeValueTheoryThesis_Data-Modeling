from bs4 import BeautifulSoup
import pandas as pd

# Load the HTML content from file
with open('/Users/megancoyne/school/thesis/bonds_data.html', 'r', encoding='utf-8') as file:
    soup = BeautifulSoup(file, 'html.parser')

# Find the table by its ID
table = soup.find('table', {'id': 'lvRollup'})

# Extract table rows
rows = []
for tr in table.find_all('tr'):
    tds = tr.find_all('td')
    if len(tds) == 6:
        row = [td.get_text(strip=True).replace(',', '') for td in tds[1:]]
        rows.append(row)

# Create a DataFrame
df = pd.DataFrame(rows, columns=[
    'Trade Date', 
    'High/Low Price (%)', 
    'High/Low Yield (%)', 
    'Trade Count', 
    'Total Trade Amount ($)'
])

# Convert numeric columns
df['Trade Count'] = pd.to_numeric(df['Trade Count'])
df['Total Trade Amount ($)'] = pd.to_numeric(df['Total Trade Amount ($)'])

# Save to CSV
output_path = '/Users/megancoyne/school/thesis/detroit_bonds_data.csv'
df.to_csv(output_path, index=False)
print(f"✅ Data saved to {output_path}")
