/**
 * main.jsx
 * 
 * Entry point for the Vite-React frontend dashboard application.
 * Mounts the main App component inside the index.html root DOM element 
 * with StrictMode enabled for best-practice checks.
 */

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

// Select the root DOM element and render the React tree
createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
