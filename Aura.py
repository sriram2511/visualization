import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from plot import plot_distribution,plot_finalscore_distribution
sns.set(style='darkgrid',palette = 'Set2')
import hmac
import streamlit as st




# 
# Apply the orange theme
#st.markdown(orange_theme, unsafe_allow_html=True)
# Load data
df = pd.read_csv('combined3jan29.csv')

st.title('Weld Process Distribution Analysis')

plot_distribution(df,'WELDPROCESS', 'Distribution of  Weld Processes')
st.pyplot(plt)

selected_weld_process = st.selectbox('Select Weld Process', ['GMAW', 'SMAW', 'GTAW', 'FCAW'], index=1)

st.write(f"Selected Weld Process: {selected_weld_process}")
gmaw_df = df[df['WELDPROCESS'] == selected_weld_process]
plot_distribution(gmaw_df, 'JOINTTYPE', f'Distribution of jointype (BUTT, TEE, LAP) in {selected_weld_process}')
st.pyplot(plt)

selected_joint_type = st.selectbox('Select Joint type', ['BUTT', 'TEE', 'LAP',], index=1)
st.write(f"Selected Weld Process: {selected_joint_type}")
gmaw_butt_df = gmaw_df[gmaw_df['JOINTTYPE'] == selected_joint_type]

plot_distribution(gmaw_butt_df,'WELDPOSITION',f'Distribution of jointype (BUTT, TEE, LAP) in {selected_weld_process}and in  {selected_joint_type}')
st.pyplot(plt)

a = gmaw_butt_df['WELDPOSITION'].unique()

selected_joint_position_type = st.selectbox('Select Joint WELDPOSITION type', a, index=1)
st.write(f"Selected Weld Process: {selected_joint_position_type}")
gmaw_butt_join_df = gmaw_butt_df[gmaw_butt_df['WELDPOSITION'] == selected_joint_position_type]

# plot_distribution(gmaw_butt_join_df,'WELDPOSITION',f'Distribution of jointype (BUTT, TEE, LAP) in {selected_weld_process} and in {selected_joint_type} in {selected_joint_position_type} position ')
# st.pyplot(plt)
total_unique_students = gmaw_butt_join_df.groupby('LESSON_NUM')['USERNAME'].nunique().reset_index()
passing_students = gmaw_butt_join_df[gmaw_butt_join_df['result'] == 'pass'].groupby('LESSON_NUM')['USERNAME'].nunique().reset_index()

merged_df = pd.merge(total_unique_students, passing_students, on='LESSON_NUM', suffixes=('_total', '_pass'))
merged_df['Passing_Percentage'] = (merged_df['USERNAME_pass'] / merged_df['USERNAME_total']) * 100

plt.figure(figsize=(10, 6))
sns.barplot(data=merged_df, x='LESSON_NUM', y='USERNAME_total', color='skyblue', label='Total Students')
sns.barplot(data=merged_df, x='LESSON_NUM', y='USERNAME_pass', color='coral', label='Passing Students')

for index, row in merged_df.iterrows():
    plt.text(row.name, row['USERNAME_total'], f'{row["Passing_Percentage"]:.0f}%', color='black', ha="center")

plt.title('Total Unique Students Attempting and Passing Lessons')
plt.xlabel('Lesson Number')
plt.ylabel('Count')
plt.xticks(range(0, 15), range(1, 16))  # Set x-ticks from 1 to 15
plt.legend()
plt.tight_layout()
plt.show()
st.pyplot(plt)
