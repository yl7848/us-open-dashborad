import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="US Open · Sponsor Analytics",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.hero {
    background: linear-gradient(135deg, #0a1628 0%, #1a2f50 60%, #0f3460 100%);
    border-radius: 16px; padding: 2.5rem 2.5rem 2rem;
    margin-bottom: 1.5rem; position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; top: -60px; right: -60px;
    width: 240px; height: 240px; border-radius: 50%;
    background: rgba(34,197,94,0.08);
}
.hero-badge {
    display: inline-block; background: rgba(34,197,94,0.15);
    border: 1px solid rgba(34,197,94,0.3); color: #4ade80;
    font-size: 0.72rem; font-weight: 500; letter-spacing: 0.1em;
    text-transform: uppercase; padding: 3px 10px; border-radius: 99px; margin-bottom: 0.75rem;
}
.hero-title {
    font-family: 'DM Serif Display', serif; font-size: 2.4rem;
    color: #f0f6ff; margin: 0 0 0.3rem; letter-spacing: -0.5px; line-height: 1.15;
}
.hero-sub { color: #7fa8d4; font-size: 0.95rem; font-weight: 300; margin: 0; letter-spacing: 0.08em; text-transform: uppercase; }
.kpi-card { background: #ffffff; border: 1px solid #e8edf5; border-radius: 12px; padding: 1.2rem 1.4rem; }
.kpi-label { font-size: 0.72rem; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; color: #8a9bbf; margin-bottom: 0.4rem; }
.kpi-value { font-family: 'DM Serif Display', serif; font-size: 2rem; color: #0a1628; line-height: 1; margin-bottom: 0.2rem; }
.kpi-delta { font-size: 0.78rem; color: #22c55e; font-weight: 500; }
.kpi-delta.down { color: #ef4444; }
.section-title { font-family: 'DM Serif Display', serif; font-size: 1.3rem; color: #0a1628; margin: 0.5rem 0 0.2rem; border-left: 3px solid #2563eb; padding-left: 0.75rem; }
.section-sub { font-size: 0.82rem; color: #8a9bbf; margin: 0 0 1rem 1rem; }
[data-testid="stSidebar"] { background: #0a1628; }
.styled-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.styled-table th { background: #f4f7fc; color: #4a5e80; font-weight: 500; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.07em; padding: 0.6rem 0.8rem; border-bottom: 2px solid #e0e8f5; text-align: left; }
.styled-table td { padding: 0.65rem 0.8rem; border-bottom: 1px solid #f0f4fa; color: #1a2f50; }
.styled-table tr:hover td { background: #f8faff; }
.roi-chip { display: inline-block; padding: 2px 10px; border-radius: 99px; font-size: 0.75rem; font-weight: 500; }
.roi-high { background: #dcfce7; color: #15803d; }
.roi-mid  { background: #fef9c3; color: #854d0e; }
.roi-low  { background: #fee2e2; color: #991b1b; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv('us_open_sponsor_rich_dataset.csv')

df = load_data()

PALETTE = {
    'Rolex': '#2563eb', 'IBM': '#16a34a', 'Emirates': '#d97706',
    'Mercedes-Benz': '#7c3aed', 'American Express': '#dc2626',
    'JPMorgan': '#0891b2', 'Heineken': '#65a30d',
}

with st.sidebar:
    st.markdown("## 🎾 SNO Analytics")
    st.markdown("---")
    all_sponsors = sorted(df['sponsor'].unique())
    selected_sponsors = st.multiselect("Filter Sponsors", all_sponsors, default=all_sponsors)
    all_years = sorted(df['year'].unique())
    year_range = st.slider("Year Range", int(all_years[0]), int(all_years[-1]), (int(all_years[0]), int(all_years[-1])))
    st.markdown("---")
    st.markdown("##### ML Configuration")
    algo = st.selectbox("Algorithm", ["Random Forest", "Linear Regression"])
    features = st.multiselect(
        "Input Features",
        ['cost_million_usd','impressions','engagements','tv_viewers','social_mentions','sentiment_score'],
        default=['cost_million_usd','impressions','engagements','sentiment_score']
    )
    st.markdown("---")
    st.caption("IBM Hackathon · Team SNO · 2024")

if not selected_sponsors:
    st.warning("Please select at least one sponsor.")
    st.stop()

mask = df['sponsor'].isin(selected_sponsors) & df['year'].between(*year_range)
dff = df[mask].copy()

st.markdown("""
<div class="hero">
    <div class="hero-badge">US Open · Sponsorship Intelligence</div>
    <div class="hero-title">Sponsor Performance<br>Dashboard</div>
    <p class="hero-sub">2018 – 2023 · 7 Major Sponsors · Media Value Analytics</p>
</div>
""", unsafe_allow_html=True)

total_imp   = dff['impressions'].sum()
avg_sent    = dff['sentiment_score'].mean()
total_media = dff['media_value_million_usd'].sum()
avg_roi     = (dff['media_value_million_usd'] / dff['cost_million_usd'] / 1000).mean()
yearly = dff.groupby('year')['media_value_million_usd'].sum().sort_index()
yoy = ((yearly.iloc[-1] - yearly.iloc[-2]) / yearly.iloc[-2] * 100) if len(yearly) >= 2 else 0.0

def kpi_html(label, value, delta=None):
    d = ""
    if delta is not None:
        cls = "down" if delta < 0 else ""
        sign = "▲" if delta >= 0 else "▼"
        d = f'<div class="kpi-delta {cls}">{sign} {abs(delta):.1f}% YoY</div>'
    return f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>{d}</div>'

c1, c2, c3, c4 = st.columns(4)
c1.markdown(kpi_html("Total Impressions", f"{total_imp/1e9:.2f}B"), unsafe_allow_html=True)
c2.markdown(kpi_html("Avg Sentiment Score", f"{avg_sent:.2f}"), unsafe_allow_html=True)
c3.markdown(kpi_html("Total Media Value", f"${total_media:,.0f}M", delta=yoy), unsafe_allow_html=True)
c4.markdown(kpi_html("Avg ROI Multiple", f"×{avg_roi:.0f}"), unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "📊 Comparison", "🤖 Prediction Model", "📋 Raw Data"])

with tab1:
    st.markdown('<div class="section-title">Media Value Trend</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Annual media value by sponsor (USD million)</div>', unsafe_allow_html=True)
    fig_line = go.Figure()
    for s in selected_sponsors:
        sdf = dff[dff['sponsor']==s].sort_values('year')
        clr = PALETTE.get(s,'#888')
        fig_line.add_trace(go.Scatter(
            x=sdf['year'], y=sdf['media_value_million_usd'],
            mode='lines+markers', name=s,
            line=dict(color=clr, width=2.5),
            marker=dict(size=8, color=clr, line=dict(color='white',width=1.5)),
            hovertemplate=f"<b>{s}</b><br>Year: %{{x}}<br>Media Value: $%{{y:.0f}}M<extra></extra>"
        ))
    fig_line.update_layout(
        plot_bgcolor='white', paper_bgcolor='white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        margin=dict(l=0,r=0,t=40,b=0), height=380,
        xaxis=dict(showgrid=False, tickvals=list(range(year_range[0], year_range[1]+1))),
        yaxis=dict(showgrid=True, gridcolor='#f0f4fa', title='Media Value ($M)'),
        font=dict(family='DM Sans', size=12), hovermode='x unified'
    )
    st.plotly_chart(fig_line, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-title">Sentiment Score Trend</div>', unsafe_allow_html=True)
        fig_s = go.Figure()
        for s in selected_sponsors:
            sdf = dff[dff['sponsor']==s].sort_values('year')
            clr = PALETTE.get(s,'#888')
            fig_s.add_trace(go.Scatter(
                x=sdf['year'], y=sdf['sentiment_score'],
                mode='lines+markers', name=s,
                line=dict(color=clr, width=2), marker=dict(size=7,color=clr),
                showlegend=False,
                hovertemplate=f"<b>{s}</b><br>%{{x}}: %{{y:.2f}}<extra></extra>"
            ))
        fig_s.update_layout(
            plot_bgcolor='white', paper_bgcolor='white',
            margin=dict(l=0,r=0,t=10,b=0), height=260,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#f0f4fa', range=[0.5,1.0]),
            font=dict(family='DM Sans', size=11)
        )
        st.plotly_chart(fig_s, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">Sponsorship Cost Over Time</div>', unsafe_allow_html=True)
        fig_c = go.Figure()
        for s in selected_sponsors:
            sdf = dff[dff['sponsor']==s].sort_values('year')
            fig_c.add_trace(go.Bar(
                x=sdf['year'], y=sdf['cost_million_usd'],
                name=s, marker_color=PALETTE.get(s,'#888'), showlegend=False,
                hovertemplate=f"<b>{s}</b><br>%{{x}}: $%{{y:.1f}}M<extra></extra>"
            ))
        fig_c.update_layout(
            barmode='group', plot_bgcolor='white', paper_bgcolor='white',
            margin=dict(l=0,r=0,t=10,b=0), height=260,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#f0f4fa', title='Sponsorship Cost ($M)'),
            font=dict(family='DM Sans', size=11)
        )
        st.plotly_chart(fig_c, use_container_width=True)

with tab2:
    st.markdown('<div class="section-title">Sponsor Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">All figures are averages over the selected years</div>', unsafe_allow_html=True)
    agg = dff.groupby('sponsor').agg(
        avg_cost=('cost_million_usd','mean'), avg_media=('media_value_million_usd','mean'),
        avg_sent=('sentiment_score','mean'), avg_imp=('impressions','mean'),
        avg_eng=('engagements','mean'), avg_tv=('tv_viewers','mean'), avg_soc=('social_mentions','mean'),
    ).reset_index()
    agg['roi'] = agg['avg_media'] / agg['avg_cost'] / 1000

    fig_b = px.scatter(
        agg, x='avg_cost', y='avg_media', size='avg_imp', color='sponsor',
        color_discrete_map=PALETTE, hover_name='sponsor', size_max=55,
        labels={'avg_cost':'Avg Cost ($M)', 'avg_media':'Avg Media Value ($M)'},
        custom_data=['roi','avg_sent']
    )
    fig_b.update_traces(hovertemplate="<b>%{hovertext}</b><br>Cost: $%{x:.1f}M<br>Media Value: $%{y:.0f}M<br>ROI: ×%{customdata[0]:.0f}<br>Sentiment: %{customdata[1]:.2f}<extra></extra>")
    fig_b.update_layout(
        plot_bgcolor='white', paper_bgcolor='white',
        font=dict(family='DM Sans', size=12), height=380,
        margin=dict(l=0,r=0,t=10,b=0),
        xaxis=dict(showgrid=True, gridcolor='#f0f4fa'),
        yaxis=dict(showgrid=True, gridcolor='#f0f4fa'), legend=dict(title=None)
    )
    st.plotly_chart(fig_b, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">Multi-Dimension Radar</div>', unsafe_allow_html=True)
        cats = ['Impressions','Engagements','TV Viewers','Social Mentions','Sentiment']
        norm_cols = ['avg_imp','avg_eng','avg_tv','avg_soc','avg_sent']
        fig_r = go.Figure()
        for _, row in agg.iterrows():
            vals = [row[c] for c in norm_cols]
            maxv = [agg[c].max() for c in norm_cols]
            normd = [v/m for v,m in zip(vals,maxv)]
            clr = PALETTE.get(row['sponsor'],'#888')
            fig_r.add_trace(go.Scatterpolar(
                r=normd+[normd[0]], theta=cats+[cats[0]],
                fill='toself', name=row['sponsor'], line_color=clr, opacity=0.8
            ))
        fig_r.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,1], tickfont=dict(size=9)), angularaxis=dict(tickfont=dict(size=11))),
            showlegend=True, legend=dict(orientation='h', y=-0.15),
            paper_bgcolor='white', font=dict(family='DM Sans', size=11),
            height=340, margin=dict(l=20,r=20,t=20,b=60)
        )
        st.plotly_chart(fig_r, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">ROI Multiple Ranking</div>', unsafe_allow_html=True)
        agg_s = agg.sort_values('roi', ascending=True)
        fig_roi = go.Figure(go.Bar(
            x=agg_s['roi'], y=agg_s['sponsor'], orientation='h',
            marker_color=[PALETTE.get(s,'#888') for s in agg_s['sponsor']],
            text=[f'×{v:.0f}' for v in agg_s['roi']], textposition='outside',
            hovertemplate="<b>%{y}</b><br>ROI: ×%{x:.0f}<extra></extra>"
        ))
        fig_roi.update_layout(
            plot_bgcolor='white', paper_bgcolor='white', font=dict(family='DM Sans', size=12),
            height=340, margin=dict(l=0,r=50,t=10,b=0),
            xaxis=dict(showgrid=True, gridcolor='#f0f4fa', title='ROI Multiple'),
            yaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig_roi, use_container_width=True)

    st.markdown('<div class="section-title">Overall Leaderboard</div>', unsafe_allow_html=True)
    agg_d = agg.sort_values('roi', ascending=False)

    def roi_chip(v):
        if v >= 200: return f'<span class="roi-chip roi-high">×{v:.0f} Excellent</span>'
        if v >= 150: return f'<span class="roi-chip roi-mid">×{v:.0f} Good</span>'
        return f'<span class="roi-chip roi-low">×{v:.0f} Low</span>'

    rows_html = ""
    for _, r in agg_d.iterrows():
        clr = PALETTE.get(r['sponsor'],'#888')
        rows_html += f"""<tr>
            <td><span style="display:inline-flex;align-items:center;gap:8px;">
                <span style="width:10px;height:10px;border-radius:50%;background:{clr};display:inline-block;"></span>
                <b>{r['sponsor']}</b></span></td>
            <td>${r['avg_cost']:.1f}M</td><td>${r['avg_media']:.0f}M</td>
            <td>{roi_chip(r['roi'])}</td><td>{r['avg_sent']:.2f}</td><td>{r['avg_imp']/1e6:.1f}M</td>
        </tr>"""
    st.markdown(f"""
    <table class="styled-table"><thead><tr>
        <th>Sponsor</th><th>Avg Cost</th><th>Avg Media Value</th><th>ROI</th><th>Sentiment</th><th>Avg Impressions</th>
    </tr></thead><tbody>{rows_html}</tbody></table>
    """, unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="section-title">Media Value Prediction Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Machine learning model trained on historical sponsorship data</div>', unsafe_allow_html=True)

    if not features:
        st.warning("Please select at least one input feature in the sidebar.")
    elif len(dff) < 10:
        st.warning("Not enough data. Please expand your filters.")
    else:
        target = 'media_value_million_usd'
        df_ml = pd.get_dummies(dff, columns=['sponsor'], drop_first=True)
        feat_cols = [c for c in features if c in df_ml.columns]
        if feat_cols:
            X = df_ml[feat_cols]
            y = df_ml[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_train)
            X_te_s = scaler.transform(X_test)
            model = RandomForestRegressor(n_estimators=100, random_state=42) if algo == "Random Forest" else LinearRegression()
            model.fit(X_tr_s, y_train)
            y_pred = model.predict(X_te_s)
            score = r2_score(y_test, y_pred)
            mae   = mean_absolute_error(y_test, y_pred)

            m1, m2, m3 = st.columns(3)
            m1.metric("R² Score", f"{score:.4f}", help="Closer to 1.0 is better")
            m2.metric("MAE", f"{mae:.2f}M")
            m3.metric("Test Samples", len(y_test))

            col_p, col_f = st.columns([2,1])
            with col_p:
                mn, mx = float(y.min()), float(y.max())
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(
                    x=y_test, y=y_pred, mode='markers',
                    marker=dict(color='#2563eb', size=9, opacity=0.7, line=dict(color='white',width=1)),
                    name='Predictions',
                    hovertemplate="Actual: $%{x:.0f}M<br>Predicted: $%{y:.0f}M<extra></extra>"
                ))
                fig_pred.add_trace(go.Scatter(
                    x=[mn,mx], y=[mn,mx], mode='lines',
                    line=dict(color='#ef4444', dash='dash', width=1.5), name='Perfect Fit'
                ))
                fig_pred.update_layout(
                    plot_bgcolor='white', paper_bgcolor='white',
                    xaxis=dict(title='Actual Media Value ($M)', showgrid=True, gridcolor='#f0f4fa'),
                    yaxis=dict(title='Predicted Media Value ($M)', showgrid=True, gridcolor='#f0f4fa'),
                    legend=dict(orientation='h', y=1.05),
                    margin=dict(l=0,r=0,t=30,b=0), height=360,
                    font=dict(family='DM Sans', size=12)
                )
                st.plotly_chart(fig_pred, use_container_width=True)

            with col_f:
                if hasattr(model, 'feature_importances_'):
                    st.markdown('<div class="section-title" style="font-size:1rem;">Feature Importance</div>', unsafe_allow_html=True)
                    fi = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=True)
                    fig_fi = go.Figure(go.Bar(
                        x=fi.values, y=fi.index, orientation='h',
                        marker_color='#2563eb',
                        text=[f'{v:.2f}' for v in fi.values], textposition='outside'
                    ))
                    fig_fi.update_layout(
                        plot_bgcolor='white', paper_bgcolor='white',
                        margin=dict(l=0,r=50,t=10,b=0), height=360,
                        xaxis=dict(showgrid=True, gridcolor='#f0f4fa'),
                        yaxis=dict(showgrid=False), font=dict(family='DM Sans', size=11)
                    )
                    st.plotly_chart(fig_fi, use_container_width=True)

            st.markdown("---")
            st.markdown('<div class="section-title">Live Prediction</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">Enter values below to predict media value</div>', unsafe_allow_html=True)
            input_cols_ui = st.columns(len(feat_cols))
            input_vals = []
            for i, feat in enumerate(feat_cols):
                val = input_cols_ui[i].number_input(feat.replace('_',' ').title(), value=float(X[feat].mean()), format="%.2f")
                input_vals.append(val)

            if st.button("🔮 Predict", type="primary"):
                scaled = scaler.transform([input_vals])
                prediction = model.predict(scaled)[0]
                st.success(f"**Predicted Media Value: ${prediction:,.2f}M**")
                col_x, col_y = st.columns(2)
                col_x.metric("Prediction", f"${prediction:,.1f}M")
                if 'cost_million_usd' in feat_cols:
                    cost_val = input_vals[feat_cols.index('cost_million_usd')]
                    col_y.metric("Estimated ROI", f"×{prediction/cost_val/1000:.0f}")

with tab4:
    st.markdown('<div class="section-title">Raw Data</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Full dataset based on current filters</div>', unsafe_allow_html=True)
    col_s, _ = st.columns([1,3])
    with col_s:
        sort_by = st.selectbox("Sort by", ['year','media_value_million_usd','cost_million_usd','sentiment_score','impressions'])
        asc = st.radio("Order", ["Descending","Ascending"]) == "Ascending"
    display_df = dff.sort_values(sort_by, ascending=asc).reset_index(drop=True)
    display_df.columns = [c.replace('_',' ').title() for c in display_df.columns]
    st.dataframe(display_df, use_container_width=True, height=480)
    csv = dff.to_csv(index=False).encode('utf-8')
    st.download_button("⬇ Download CSV", data=csv, file_name='us_open_sponsors_filtered.csv', mime='text/csv')
