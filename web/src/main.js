import './main.css'
import axios from 'axios'

const app = document.getElementById('app')
app.innerHTML = `
  <div class="min-h-screen p-6">
    <div class="max-w-3xl mx-auto space-y-6">
      <h1 class="text-2xl font-semibold">AppMarketing Speaker Overlay</h1>
      <p class="text-slate-600">Upload a video. Overlays are preset and backgrounds are already removed. Settings are fixed: bottom-center position, 0.35 width ratio, CPU (libx264) encoder.</p>

      <form id="form" class="space-y-4">
        <div>
          <label class="block text-sm font-medium">Video (mp4/mov)</label>
          <input id="video" type="file" accept="video/*" class="mt-1 block w-full" required />
        </div>
        

        <button id="submit" class="px-4 py-2 bg-blue-600 text-white rounded">Run</button>
      </form>

      <div id="status" class="text-sm text-slate-600 whitespace-pre-wrap"></div>
      <video id="result" class="w-full mt-4" controls></video>
    </div>
  </div>
`

const form = document.getElementById('form')
const statusEl = document.getElementById('status')
const result = document.getElementById('result')

function genJobId() {
  return Math.random().toString(16).slice(2) + Date.now().toString(16)
}

let poller = null

form.addEventListener('submit', async (e) => {
  e.preventDefault()
  statusEl.textContent = 'Uploading and processing…\n'
  result.src = ''

  const data = new FormData()
  const video = document.getElementById('video').files[0]

  data.append('video', video)
  // Overlays are preset; no overlay uploads

  const jobId = genJobId()
  data.append('jobId', jobId)

  // Start log polling
  const poll = async () => {
    try {
      const r = await fetch(`/api/logs/${jobId}`)
      if (r.ok) {
        const j = await r.json()
        statusEl.textContent = j.text
      }
    } catch {}
  }
  poller = setInterval(poll, 1000)

  try {
    const resp = await axios.post('/api/process', data, {
      responseType: 'blob',
      headers: { 'Accept': 'video/mp4' },
      timeout: 1000 * 60 * 60, // up to 1h for long videos
    })
    const url = URL.createObjectURL(resp.data)
    result.src = url
    await poll()
    statusEl.textContent += '\nDone.'
  } catch (err) {
    console.error(err)
    await poll()
    statusEl.textContent += '\nError: ' + (err?.response?.data?.error || err.message)
  }
  if (poller) clearInterval(poller)
})


