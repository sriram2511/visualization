import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import skew
import hmac
st.set_option('deprecation.showPyplotGlobalUse', False)
sns.set(style = 'darkgrid',palette = 'Set3')
def check_password():
    """Returns `True` if the user had a correct password."""

    def login_form():
        """Form with widgets to collect user information"""
        with st.form("Credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] in st.secrets[
            "passwords"
        ] and hmac.compare_digest(
            st.session_state["password"],
            st.secrets.passwords[st.session_state["username"]],
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the username or password.
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    # Return True if the username + password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state:
        st.error("😕 User not known or password incorrect")
    return False


if not check_password():
    st.stop()

# Load data
df = pd.read_csv('chroma.csv', low_memory=False)
df = df[~(df['distance'] + df['speed'] + df['rotational'] +
          df['horizontal'] + df['vertical'] + df['coverage'] == 0)]

st.title('Chroma Analysis')
ax = sns.countplot(x = 'process', data = df)
ax.bar_label(ax.containers[0])
st.pyplot()
selected_process = st.selectbox("Choose any of the following", df['process'].unique())
if selected_process in df['process'].unique():
    selected_process_df = df[df['process'] == selected_process]
else:
    selected_process_df = df
ax = sns.countplot(x = 'workpiece', data = selected_process_df)
ax.bar_label(ax.containers[0])
plt.xticks(rotation = 90)
st.pyplot()

workpiece = selected_process_df['workpiece'].unique()
selected_workpiece = st.selectbox('Select Workpiece', workpiece)
if selected_workpiece in workpiece:
    workpiece_df = selected_process_df[selected_process_df['workpiece'] == selected_workpiece]
else:
    workpiece_df = selected_process_df
ax = sns.countplot(x = 'stage', data = workpiece_df,order = ['Primer','Base Coat','Top Coat'])
ax.bar_label(ax.containers[0])
st.pyplot()

stage =  ['Primer','Base Coat','Top Coat']
selected_stage = st.selectbox("Select Stage",stage)
stage_df = workpiece_df[workpiece_df['stage'] == selected_stage]

result_counts = stage_df.groupby(['USERNAME', 'result']).size().reset_index(name='count')
result_pivot = result_counts.pivot(index='USERNAME', columns='result', values='count').fillna(0)
result_pivot['total'] = result_pivot['pass'] + result_pivot['fail']
result_pivot.reset_index(inplace=True)
grouped_df = stage_df.groupby('USERNAME')
fail_count_before_pass = {}

for name, group in grouped_df:
    fail_count = 0
    found_pass = False
    for index, row in group.iterrows():
        if row['result'] == 'fail':
            fail_count += 1
        elif row['result'] == 'pass' and not found_pass:
            fail_count_before_pass[name] = fail_count
            found_pass = True
result_df = pd.DataFrame(list(fail_count_before_pass.items()), columns=['USERNAME', 'FailCountBeforePass'])


st.subheader('Pie Chart and Distribution Plots')

    # Set up the figure with a 1-row, 2-column grid
fig, axs = plt.subplots(1, 2, figsize=(15, 8))

    # Plot 1: Pie Chart (takes half the width)
labels = ['Fail', 'Pass']
sizes = [result_pivot[result_pivot['pass'] == 0].shape[0], result_pivot[result_pivot['pass'] >= 1].shape[0]]
axs[0].pie(sizes, labels=labels, colors=['#ff9999', '#66b3ff'], autopct='%1.1f%%', startangle=90)
axs[0].legend(['Fail', 'Pass'])
axs[0].set_title('Pie Chart')

    # Plot 2: Histogram
sns.histplot(x=result_df['FailCountBeforePass'], kde=True, fill=True, ax=axs[1])
axs[1].set_title('Histogram')


st.pyplot(fig)

    # Skewness Information and Outliers
st.subheader('Skewness Information and Outliers')
skewness_value = skew(result_df['FailCountBeforePass'])
q3, q1 = np.percentile(result_df['FailCountBeforePass'], [75, 25])
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = [x for x in result_df['FailCountBeforePass'] if x < lower_bound or x > upper_bound]
left_tail_outliers = [x for x in result_df['FailCountBeforePass'] if x < lower_bound]
right_tail_outliers = [x for x in result_df['FailCountBeforePass'] if x > upper_bound]

if skewness_value > 0:
    st.text("The distribution is positively skewed (right-skewed).")
    st.text("Possible outliers: {}".format(right_tail_outliers))
elif skewness_value < 0:
    st.text("The distribution is negatively skewed (left-skewed).")
    st.text("Possible outliers: {}".format(left_tail_outliers))
else:
    st.text("The distribution is perfectly symmetrical.")


st.text("Most users have low fail counts before the 1st pass.")
st.text("Most no of students have count of {}".format(result_df['FailCountBeforePass'].mode().values[0]))
st.text(f"Average attempts before passing is {int(result_df['FailCountBeforePass'].mean())}")
result_df_no_outliers = result_df[(result_df['FailCountBeforePass'] >= lower_bound) & (result_df['FailCountBeforePass'] <= upper_bound)]
st.text(f"After Removing Outliets Average attempts before passing is {int(result_df_no_outliers['FailCountBeforePass'].mean())}")
