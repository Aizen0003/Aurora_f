import pandas as pd, json

print('\n--- ITERATION 1 DELIVERABLES ---')

# 10. experiment_results.csv
try:
    df = pd.read_csv('data/sample/experiment_results_sample.csv')
    print(f'\n[10] experiment_results.csv: OK ({len(df)} rows)')
    print(f'     Columns: {list(df.columns)}')
    ps_cols = ['template_id','segment_id','lifecycle_stage','goal','theme','notification_window',
               'total_sends','total_opens','total_engagements','ctr','engagement_rate',
               'uninstall_rate','performance_status']
    missing = [c for c in ps_cols if c not in df.columns]
    print(f'     PS required cols missing: {missing if missing else "NONE - all present"}')
except Exception as e:
    print(f'\n[10] experiment_results.csv: FAIL - {e}')

# 11. learning_delta_report.csv
try:
    df = pd.read_csv('data/output/learning_delta_report.csv')
    print(f'\n[11] learning_delta_report.csv: OK ({len(df)} rows)')
    print(f'     Columns: {list(df.columns)}')
    ps_cols = ['entity_type','entity_id','change_type','metric_trigger','before_value','after_value','explanation']
    missing = [c for c in ps_cols if c not in df.columns]
    print(f'     PS required cols missing: {missing if missing else "NONE - all present"}')
except Exception as e:
    print(f'\n[11] learning_delta_report.csv: FAIL - {e}')

# 12. Improved files
for fn in ['message_templates_improved.csv', 'timing_recommendations_improved.csv', 'user_notification_schedule_improved.csv']:
    try:
        df = pd.read_csv(f'data/output/{fn}')
        print(f'\n[12] {fn}: OK ({len(df)} rows)')
    except Exception as e:
        print(f'\n[12] {fn}: FAIL - {e}')

# 13. README.txt
try:
    with open('README.txt', 'r') as f:
        content = f.read()
    words = len(content.split())
    print(f'\n[13] README.txt: OK ({words} words)')
    limit_ok = "PASS" if words <= 500 else "FAIL - TOO LONG"
    print(f'     PS limit: <=500 words. {limit_ok}')
except Exception as e:
    print(f'\n[13] README.txt: FAIL - {e}')

# Schedule notif slots
sched = pd.read_csv('data/output/user_notification_schedule.csv')
notif_cols = [c for c in sched.columns if c.startswith('notif_') and c.endswith('_template_id')]
print(f'\n--- SCHEDULE ANALYSIS ---')
print(f'Max notification slots: {len(notif_cols)} (PS says 3-9)')
for i in range(1, 10):
    col = f'notif_{i}_template_id'
    if col in sched.columns:
        non_null = sched[col].notna().sum()
        print(f'  notif_{i}: EXISTS ({non_null} non-null)')
    else:
        print(f'  notif_{i}: MISSING')

# Check user_segments has PS-required columns
print('\n--- USER SEGMENTS COLUMNS CHECK ---')
seg = pd.read_csv('data/output/user_segments.csv')
ps_seg_needs = ['gamification_propensity', 'social_propensity', 'activeness', 'churn_risk']
for c in ps_seg_needs:
    if c in seg.columns:
        print(f'  {c}: OK')
    else:
        print(f'  {c}: MISSING!')

# Check segment count
n_segments = seg['segment_id'].nunique()
print(f'  Segment count: {n_segments} (PS requires 6-12)')

# Check communication_themes has tone info
print('\n--- COMM THEMES COLUMNS CHECK ---')
themes = pd.read_csv('data/output/communication_themes.csv')
print(f'  Columns: {list(themes.columns)}')
# PS says: Primary/secondary themes, tone preferences, hooks per segment
# Missing tone_preferences? hooks?

# Check segment_goals has day-on-day
print('\n--- SEGMENT GOALS CHECK ---')
goals = pd.read_csv('data/output/segment_goals.csv')
print(f'  Columns: {list(goals.columns)}')
print(f'  Has lifecycle_day: {"lifecycle_day" in goals.columns}')
days_covered = goals['lifecycle_day'].nunique() if 'lifecycle_day' in goals.columns else 0
print(f'  Days covered: {days_covered}')
