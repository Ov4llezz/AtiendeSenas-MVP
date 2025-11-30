/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'totem-primary': '#2563eb',
        'totem-success': '#10b981',
        'totem-warning': '#f59e0b',
        'totem-danger': '#ef4444',
      }
    },
  },
  plugins: [],
}
