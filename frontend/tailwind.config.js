/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        fleet: {
          bg: "#F4F5F7",
          panel: "#FFFFFF",
          text: "#202833",
          muted: "#697586",
          border: "#DDE2E8",
          primary: "#2457A7",
          primaryHover: "#1D478A",
          normal: "#287D52",
          warning: "#B7791F",
          critical: "#C43D3D",
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
