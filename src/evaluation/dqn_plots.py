"""Seven Stage-5 figures; fixed first full June week, no selection by performance."""
import pandas as pd
import numpy as np

def make_dqn_plots(history,frames,comparison,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import StrMethodFormatter
    from .compare_baselines import select_example_week
    destination.mkdir(parents=True,exist_ok=True)
    colors={'No Battery':'#52677D','Rule-Based':'#00867D','Q-Learning':'#B77723','DQN':'#6B4BA5'}
    dqn=frames['DQN'];week=select_example_week(dqn.loc[dqn.timestamp.dt.month==6])
    start=week.timestamp.iloc[0];end=week.timestamp.iloc[-1]+pd.Timedelta(hours=1)
    weeknote=f'{start:%d %b}–{end-pd.Timedelta(days=1):%d %b %Y} · First full June week · No weekly SOC reset'
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
       'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'axes.titleweight':'bold',
       'figure.facecolor':'white','axes.axisbelow':True}):
        def save(fig,name,note):
            fig.text(.09,.025,note,fontsize=9,color='#526174')
            fig.savefig(destination/name,dpi=180,facecolor='white');plt.close(fig)
        def base(title,ylabel):
            fig,ax=plt.subplots(figsize=(11,5.5))
            fig.subplots_adjust(left=.11,right=.97,bottom=.20,top=.86)
            ax.set_title(title,loc='left',pad=16,fontsize=16);ax.set_ylabel(ylabel)
            return fig,ax
        def dates(ax):
            ax.set_xlim(start,end);ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%a\n%d %b'))
        fig,ax=base('DQN training reward · 2023','Episode reward (€)')
        ax.plot(history.episode,history.cumulative_reward,color='#BCAED4',lw=1,label='Exploratory episode')
        ax.plot(history.episode,history.cumulative_reward.rolling(5,min_periods=1).mean(),color=colors['DQN'],lw=2,label='Trailing 5-episode mean')
        ax.set_xlabel('Episode');ax.legend(frameon=False)
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        save(fig,'stage5_training_reward.png','Fixed final episode · Reward includes unchanged €0.01/kWh throughput penalty\nExploratory training reward is not greedy validation/test performance.')
        fig,ax=base('DQN training loss · 2023','Mean minibatch Huber loss')
        ax.plot(history.episode,history.mean_training_loss,color='#BCAED4',lw=1,label='Episode mean')
        ax.plot(history.episode,history.mean_training_loss.rolling(5,min_periods=1).mean(),color=colors['DQN'],lw=2,label='Trailing 5-episode mean')
        ax.set_xlabel('Episode');ax.legend(frameon=False)
        save(fig,'stage5_training_loss.png','Loss aggregated by episode · Uniform replay · Target network synchronized every 1,000 environment steps')
        fig,ax=base('Electricity purchasing cost · 2025','Cumulative raw cost (€)')
        for name,frame in frames.items():
            cost=frame.grid_cost_eur.cumsum()
            ax.plot(frame.timestamp+pd.Timedelta(hours=1),cost,color=colors[name],lw=1.9,label=f'{name} · €{cost.iloc[-1]:,.2f}')
        loc=mdates.AutoDateLocator(minticks=5,maxticks=8);ax.xaxis.set_major_locator(loc);ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'));ax.legend(frameon=False,loc='upper left')
        save(fig,'stage5_test_cumulative_cost.png','Same full-year 2025 test and dispatch · Raw bill excludes terminal valuation and cycling penalty')
        for kind in ['soc','actions']:
            fig,ax=base('DQN battery state of charge' if kind=='soc' else 'DQN requested actions','SOC (%)' if kind=='soc' else 'Requested action')
            if kind=='soc':
                ax.step([start,*(week.timestamp+pd.Timedelta(hours=1))],[week.soc_before_kwh.iloc[0],*(week.soc_fraction*100)],where='post',color=colors['DQN'],lw=1.5)
                for y in [10,95]:ax.axhline(y,color='#526174',ls='--',lw=1)
                ax.set_ylim(0,100)
            else:
                ax.step([*week.timestamp,end],[*week.action,week.action.iloc[-1]],where='post',color=colors['DQN'],lw=1.5)
                ax.set_yticks([-1,0,1],['Discharge (−1)','Hold (0)','Charge (+1)']);ax.set_ylim(-1.3,1.3)
                fig.subplots_adjust(left=.18)
            dates(ax)
            save(fig,f'stage5_dqn_{kind}_week.png',weeknote+'\nRequests may produce zero flow at SOC boundaries or with no residual load.')
        fig,axes=plt.subplots(3,1,sharex=True,figsize=(11,8),height_ratios=[2,1,1])
        fig.subplots_adjust(left=.11,right=.97,bottom=.14,top=.91,hspace=.20)
        axes[0].set_title('DQN solar-storage behavior · sunny test week',loc='left',pad=16,fontsize=16)
        axes[0].plot(week.timestamp,week.demand_kwh,color='#263F58',label='Demand')
        axes[0].plot(week.timestamp,week.pv_generation_kwh,color='#00867D',label='PV generation')
        axes[0].fill_between(week.timestamp,week.demand_kwh,week.pv_generation_kwh,where=week.pv_generation_kwh>week.demand_kwh,color='#72B39B',alpha=.35,label='PV surplus')
        axes[0].set_ylabel('Hourly energy (kWh)');axes[0].legend(frameon=False,ncol=3,loc='upper left')
        axes[1].fill_between(week.timestamp,0,(week.pv_generation_kwh-week.demand_kwh).clip(lower=0),color='#72B39B',alpha=.35,label='Available surplus')
        axes[1].plot(week.timestamp,week.pv_to_battery_kwh,color=colors['DQN'],lw=1.7,label='Actual PV into battery')
        axes[1].set_ylabel('PV energy (kWh)');axes[1].legend(frameon=False,ncol=2,loc='upper left')
        axes[2].step(week.timestamp+pd.Timedelta(hours=1),week.soc_fraction*100,color=colors['DQN'],where='post')
        axes[2].set_ylim(0,100);axes[2].set_ylabel('SOC (%)');dates(axes[2])
        save(fig,'stage5_pv_capture_week.png',weeknote+'\nPV capture measures actual flow into storage before conversion losses.')
        fig,ax=plt.subplots(figsize=(12,4.5));ax.axis('off')
        fig.subplots_adjust(left=.04,right=.98,bottom=.22,top=.78)
        ax.set_title('2025 business comparison · benefits and tradeoffs',loc='left',pad=22,fontsize=16,fontweight='bold')
        rows=[]
        for r in comparison.itertuples():
            rows.append([r.Strategy,f'€{r.terminal_soc_adjusted_cost_eur:,.2f}',f'{r.adjusted_cost_savings_pct:.3f}%',
                         f'{r.pv_to_battery_kwh:,.1f}',f'{r.efc_per_day:.4f}',f'{r.peak_grid_import_kwh_per_hour:.1f}'])
        tab=ax.table(cellText=rows,colLabels=['Controller','SOC-adjusted\ncost','Adjusted\nsavings','PV to battery\n(kWh)','EFC / day','Peak import\n(kW)'],cellLoc='center',loc='center',colWidths=[.16,.20,.15,.18,.14,.17])
        tab.auto_set_font_size(False);tab.set_fontsize(11);tab.scale(1,2.4)
        for (r,c),cell in tab.get_celld().items():
            cell.set_edgecolor('white')
            cell.set_facecolor('#E8EDF3' if r==0 else ('#F2EEF8' if r==4 else '#F7F9FB'))
            if r==0:cell.set_text_props(weight='bold')
        save(fig,'stage5_business_comparison.png','Adjusted savings reference: No Battery · Inventory valued using training price and battery efficiency\nLower cost can coexist with higher grid peaks or cycling. No investment-return claim.')
