# Power BI Dashboard

This folder is where the actual `.pbix` file goes once you build it.

## How to build it
Follow the step-by-step guide: connect Power BI Desktop to your local MySQL
`churnguard` database, then paste in the queries from
`../sql/03_dashboard_queries.sql` one at a time to load each panel's data.
Build four Card visuals for the KPIs, a bar chart for segment churn rate, a
line chart for the inactivity trend, and a table for the retention watchlist.

## Once it's built
1. Save the file here as `churnguard_dashboard.pbix`
2. Take a screenshot of the finished dashboard and save it as
   `dashboard_screenshot.png` in this same folder
3. Add the screenshot to the main `README.md` under the Results section, e.g.:
   ```markdown
   ![Power BI Dashboard](powerbi/dashboard_screenshot.png)
   ```

This folder is intentionally **not** gitignored — unlike `outputs/`, the
`.pbix` file and screenshot here are meant to be committed to the repo so
anyone viewing it on GitHub can see the actual dashboard.
