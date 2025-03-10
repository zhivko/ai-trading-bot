from flask import Flask, request, render_template_string
import glob
import pandas as pd
import json
import os
from datetime import timezone  # Import timezone for UTC handling

app = Flask(__name__)

def get_episode_files():
    print(f"Current working directory: {os.getcwd()}")
    files = sorted(glob.glob("episode_*.csv"), key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)
    print(f"Found files: {files}")
    return files

@app.route("/", methods=["GET"])
def index():
    episode_files = get_episode_files()
    selected_episode = request.args.get("episode")
    
    if not selected_episode and episode_files:
        selected_episode = episode_files[0]
    
    try:
        offset = int(request.args.get("offset", 0))
    except ValueError:
        offset = 0

    ohlc_data = []
    trade_events = []
    networth_data = []

    if selected_episode:
        trade_data = pd.read_csv(selected_episode)
        
        # Convert datetime with explicit format and handle timezones
        trade_data['datetime'] = pd.to_datetime(
            trade_data['datetime'],
            utc=True,
            format='mixed',  # Handle multiple formats
            errors='coerce'  # Convert unparseable dates to NaT
        )

        # Drop rows with invalid datetime values
        trade_data = trade_data.dropna(subset=['datetime'])
        trade_data = trade_data.sort_values("datetime")

        # Add debug check for remaining datetime issues
        if trade_data['datetime'].isna().any():
            print(f"Found {trade_data['datetime'].isna().sum()} invalid datetime entries in {selected_episode}")
        
        if not trade_data.empty:
            # Get reference time as datetime object
            reference_time = trade_data["datetime"].max()
            
            # Convert offset to integer safely
            try:
                offset = int(offset)
            except:
                offset = 0

            print(f"Reference time type: {type(reference_time)}, value: {reference_time}")
            print(f"Offset type: {type(offset)}, value: {offset}")

            display_end = reference_time - pd.Timedelta(days=offset)
            display_start = display_end - pd.Timedelta(days=20)
            
            file_path = r".\BTCUSDT.data"
            btc_data = pd.read_csv(file_path, parse_dates=["datetime"])
            
            # Check if datetime is already timezone-aware and handle accordingly
            if btc_data['datetime'].dt.tz is not None:
                # If already timezone-aware, convert to UTC if not already UTC
                if btc_data['datetime'].dt.tz != timezone.utc:
                    btc_data['datetime'] = btc_data['datetime'].dt.tz_convert(timezone.utc)
                else:
                    # Already in UTC, no conversion needed
                    pass
            else:
                # If not timezone-aware, localize to UTC
                btc_data['datetime'] = pd.to_datetime(btc_data['datetime']).dt.tz_localize(timezone.utc)
            
            btc_data.sort_values("datetime", inplace=True)
            
            plot_data = btc_data[(btc_data["datetime"] >= display_start) & (btc_data["datetime"] <= display_end)]
            if plot_data.empty:
                plot_data = btc_data.tail(20)
            
            for idx, row in plot_data.iterrows():
                ohlc_data.append({
                    "x": row["datetime"].isoformat(),
                    "o": row["open"],
                    "h": row["high"],
                    "l": row["low"],
                    "c": row["close"]
                })
            
            trade_window = trade_data[(trade_data["datetime"] >= display_start) &
                                     (trade_data["datetime"] <= display_end)]
            if trade_window.empty:
                trade_window = trade_data.tail(20)
            
            current_position = 0.0
            for idx, row in trade_window.iterrows():
                action = row["action"].lower().strip()
                
                # Get reward value and ensure it's numeric
                reward_value = pd.to_numeric(row.get("reward", 0), errors='coerce')
                
                # Check for inconsistencies in reward calculations for short positions
                if action in ['sell', 'open short', 'close_short']:
                    # For short positions, check if reward direction matches price action
                    if idx > 0 and 'price' in trade_window.columns:
                        prev_price = float(trade_window.iloc[idx-1]["price"])
                        curr_price = float(row["price"])
                        price_increased = curr_price > prev_price
                        
                        # For debugging purposes
                        if price_increased and reward_value > 0 and action == 'open short':
                            print(f"Warning: Inconsistent reward for short position at {row['datetime']}: "
                                  f"Price increased ({prev_price:.2f} to {curr_price:.2f}) but reward is positive ({reward_value:.4f})")
                
                if action in ['buy', 'open long']:
                    current_position = abs(float(row.get("quantity", 0)))
                    trade_events.append({
                        "timestamp": row["datetime"].isoformat(),
                        "action": action,
                        "price": row["price"],
                        "quantity": current_position,
                        "networth": row["networth"],
                        "reward_components": {
                            "total": reward_value
                        }
                    })
                elif action in ['sell', 'open short']:
                    current_position = -abs(float(row.get("quantity", 0)))
                    trade_events.append({
                        "timestamp": row["datetime"].isoformat(),
                        "action": action,
                        "price": row["price"],
                        "quantity": abs(current_position),
                        "networth": row["networth"],
                        "reward_components": {
                            "total": reward_value
                        }
                    })
                elif action in ['close_long', 'close_short']:
                    quantity = abs(current_position) if current_position != 0 else 0.0
                    current_position = 0.0
                    trade_events.append({
                        "timestamp": row["datetime"].isoformat(),
                        "action": action,
                        "price": row["price"],
                        "quantity": quantity,
                        "networth": row["networth"],
                        "reward_components": {
                            "total": reward_value
                        }
                    })
                else:  # 'hold' or unrecognized actions
                    trade_events.append({
                        "timestamp": row["datetime"].isoformat(),
                        "action": action,
                        "price": row["price"],
                        "quantity": 0.0,
                        "networth": row["networth"],
                        "reward_components": {
                            "total": reward_value
                        }
                    })
                
                networth_data.append({
                    "x": row["datetime"].isoformat(),
                    "y": row["networth"]
                })

        # Ensure numeric types when processing trade data
        trade_data['price'] = pd.to_numeric(trade_data['price'], errors='coerce')
        trade_data['quantity'] = pd.to_numeric(trade_data['quantity'], errors='coerce')
        trade_data['networth'] = pd.to_numeric(trade_data['networth'], errors='coerce')
        trade_data = trade_data.dropna(subset=['price', 'quantity', 'networth'])

        print(f"Processing {selected_episode} with {len(trade_data)} valid records")
        print("Sample datetimes:", trade_data['datetime'].head().dt.strftime('%Y-%m-%d %H:%M:%S%z').tolist())
    
    prev_offset = offset + 20
    next_offset = max(offset - 20, 0)

    html_template = """
    <!DOCTYPE html>
    <html>
      <head>
        <title>BTC/USDT Trading Visualization</title>
        <script src="https://cdn.plot.ly/plotly-2.18.0.min.js"></script>
        <style>
          body {
            margin: 0;
            padding: 15px;
            height: 100vh;
            overflow: hidden;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
          }
          #combinedChart {
            width: 100%;
            height: 82vh;
            flex-grow: 1;
            margin-bottom: 10px;
          }
          .nav-links {
            margin: 10px 0;
            text-align: center;
          }
          .nav-links a {
            margin: 0 10px;
            padding: 5px 15px;
            background: #f0f0f0;
            border-radius: 4px;
            text-decoration: none;
          }
          form {
            margin-bottom: 10px;
          }
          h1 {
            margin: 5px 0 15px 0;
            font-size: 1.8em;
          }
        </style>
      </head>
      <body>
        <h1>BTC/USDT Trading Visualization</h1>
        <form method="get" onchange="this.submit()">
          Select Episode:
          <select name="episode">
            {% for ep in episode_files %}
              <option value="{{ ep }}" {% if ep == selected_episode %}selected{% endif %}>{{ ep }}</option>
            {% endfor %}
          </select>
          <input type="hidden" name="offset" value="{{ offset }}">
        </form>
        
        <div class="nav-links">
          <a href="#" onclick="loadData('first')">First</a>
          <a href="#" onclick="loadData('prev')">Previous</a>
          <a href="#" onclick="loadData('next')">Next</a>
          <a href="#" onclick="loadData('latest')">Latest</a>
        </div>

        <div id="combinedChart"></div>

        <script>
          var ohlcData = {{ ohlc_data|tojson }};
          var dates = ohlcData.map(d => d.x);
          var opens = ohlcData.map(d => d.o);
          var highs = ohlcData.map(d => d.h);
          var lows = ohlcData.map(d => d.l);
          var closes = ohlcData.map(d => d.c);
          var maxPrice = Math.max(...highs);

          var tradeEvents = {{ trade_events|tojson }};
          var eventCategories = {
            'buy': { symbol: 'triangle-up', color: 'blue', name: 'Open Long' },
            'sell': { symbol: 'triangle-down', color: 'red', name: 'Open Short' },
            'close_long': { symbol: 'triangle-up-open', color: 'blue', name: 'Close Long' },
            'close_short': { symbol: 'triangle-down-open', color: 'red', name: 'Close Short' },
            'stop_loss_sideways': { symbol: 'circle-open', color: '#FF0000', name: 'Stop Loss (Sideways)' }
          };
          
          var tradeTraces = Object.entries(eventCategories).map(([action, style]) => ({
            x: tradeEvents.filter(te => te.action.toLowerCase() === action.toLowerCase()).map(d => d.timestamp),
            y: tradeEvents.filter(te => te.action.toLowerCase() === action.toLowerCase()).map(d => d.price),
            mode: 'markers',
            marker: { 
              size: d => Math.min(Math.sqrt(d.quantity) * 40 + 100, 150),
              color: style.color,
              symbol: style.symbol,
              line: { width: 2.5, color: style.color }
            },
            name: style.name,
            hoverinfo: 'text',
            text: tradeEvents.filter(te => te.action.toLowerCase() === action.toLowerCase())
              .map(d => `${style.name}<br>Price: ${Number(d.price)?.toFixed(2) || 'N/A'}<br>Qty: ${Number(d.quantity)?.toFixed(3) || '0.000'}<br>NW: ${Number(d.networth)?.toFixed(2) || 'N/A'}`)
          }));

          var networthData = {{ networth_data|tojson }};
          var netValues = networthData.map(d => d.y);
          var traceNetworth = {
            x: networthData.map(d => d.x),
            y: netValues,
            type: 'scatter',
            mode: 'lines',
            line: {color: 'black', width: 1},
            name: 'Net Worth',
            yaxis: 'y2'
          };

          var rewardComponents = {
            timestamps: [],
            total: []
          };
          
          tradeEvents.forEach(te => {
            rewardComponents.timestamps.push(te.timestamp);
            rewardComponents.total.push(te.reward_components?.total || 0);
          });

          var rewardTraces = [
            {name: 'Total Reward', y: rewardComponents.total, line: {color: '#2ca02c', width: 1}}
          ].map(trace => ({
            x: rewardComponents.timestamps,
            y: trace.y,
            type: 'scatter',
            mode: 'lines',
            name: trace.name,
            line: trace.line,
            xaxis: 'x',
            yaxis: 'y3'
          }));

          // Add horizontal reference line at y=0 for rewards subplot
          var zeroLine = {
            x: [rewardComponents.timestamps[0], rewardComponents.timestamps[rewardComponents.timestamps.length-1]],
            y: [0, 0],
            type: 'scatter',
            mode: 'lines',
            name: 'Zero Reference',
            line: {
              color: 'rgba(0, 0, 0, 0.5)',
              width: 1,
              dash: 'dash'
            },
            xaxis: 'x',
            yaxis: 'y3'
          };

          var combinedData = [
            {
              x: dates,
              open: opens,
              high: highs,
              low: lows,
              close: closes,
              type: 'candlestick',
              name: 'BTC/USDT',
              yaxis: 'y'
            },
            ...tradeTraces,
            traceNetworth,
            ...rewardTraces,
            zeroLine
          ];

          var layout = {
            title: 'BTC/USDT Trading Visualization',
            font: {
                family: 'Arial',
                size: 10,
                color: '#333'
            },
            xaxis: { 
                title: 'Date', 
                rangeslider: { visible: false },
                domain: [0, 0.95],
                titlefont: { size: 10 },
                tickfont: { size: 8 }
            },
            yaxis: {
                title: 'Price (USD)',
                domain: [0.3, 1],
                automargin: true,
                titlefont: { size: 10 },
                tickfont: { size: 8 }
            },
            yaxis2: {
                title: 'Net Worth (USD)',
                overlaying: 'y',
                side: 'right',
                domain: [0.3, 1],
                titlefont: { size: 10 },
                tickfont: { size: 8 }
            },
            yaxis3: {
                title: 'Reward Components',
                anchor: 'x',
                domain: [0, 0.3],
                showgrid: true,
                zeroline: false,
                automargin: true,
                autorange: true,
                titlefont: { size: 10 },
                tickfont: { size: 8 }
            },
            showlegend: true,
            legend: {
                x: 0.02,
                y: 1.15,
                xanchor: 'left',
                yanchor: 'top',
                font: { size: 8 },
                tracegroupgap: 3,
                itemsizing: 'constant',
                symbolwidth: 12
            },
            margin: { 
                t: 100,
                b: 40,
                l: 50,
                r: 50
            }
          };

          Plotly.newPlot('combinedChart', combinedData, layout, {
            responsive: true,
            displayModeBar: false
          });

          var episodeFiles = {{ episode_files|tojson|safe }};
          var currentIndex = episodeFiles.indexOf("{{ selected_episode|safe }}");
          if(currentIndex === -1) currentIndex = 0;

          function loadData(direction) {
            const urlParams = new URLSearchParams(window.location.search);
            let currentOffset = parseInt(urlParams.get('offset') || 0);
            let newOffset = currentOffset;
            
            switch(direction) {
              case 'prev':
                newOffset = currentOffset + 20;
                break;
              case 'next':
                newOffset = Math.max(0, currentOffset - 20);
                break;
              case 'first':
                newOffset = 1000;
                break;
              case 'latest':
                newOffset = 0;
                break;
            }
            
            const currentEpisode = "{{ selected_episode|safe }}";
            window.location.href = `?episode=${currentEpisode}&offset=${newOffset}`;
          }
        </script>
      </body>
    </html>
    """
    
    return render_template_string(html_template,
                                 episode_files=episode_files,
                                 selected_episode=selected_episode,
                                 offset=offset,
                                 prev_offset=prev_offset,
                                 next_offset=next_offset,
                                 ohlc_data=ohlc_data,
                                 trade_events=trade_events,
                                 networth_data=networth_data)

if __name__ == "__main__":
    app.run(debug=True)