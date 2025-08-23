import './main.css'
import axios from 'axios'

const app = document.getElementById('app')
app.innerHTML = `
  <div class="min-h-screen p-6">
    <div class="max-w-3xl mx-auto space-y-6">
      <h1 class="text-2xl font-semibold">AppMarketing Speaker Overlay</h1>
      <p class="text-slate-600">Upload a video and optional GIF overlays. Background removal is automatic.</p>

      <form id="form" class="space-y-4">
        <div>
          <label class="block text-sm font-medium">Video (mp4/mov)</label>
          <input id="video" type="file" accept="video/*" class="mt-1 block w-full" required />
        </div>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label class="block text-sm font-medium">Overlay A (gif)</label>
            <input id="overlayA" type="file" accept="image/gif" class="mt-1 block w-full" />
          </div>
          <div>
            <label class="block text-sm font-medium">Overlay B (gif)</label>
            <input id="overlayB" type="file" accept="image/gif" class="mt-1 block w-full" />
          </div>
        </div>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label class="block text-sm font-medium">Position</label>
            <select id="position" class="mt-1 block w-full">
              <option value="top-right">top-right</option>
              <option value="top-left">top-left</option>
              <option value="bottom-right">bottom-right</option>
              <option value="bottom-left">bottom-left</option>
            </select>
          </div>
          <div>
            <label class="block text-sm font-medium">Overlay width ratio</label>
            <input id="ratio" type="number" step="0.05" min="0.05" max="0.8" value="0.2" class="mt-1 block w-full" />
          </div>
          <div>
            <label class="block text-sm font-medium">Device</label>
            <select id="device" class="mt-1 block w-full">
              <option value="auto">auto</option>
              <option value="mps">mps</option>
              <option value="cuda">cuda</option>
              <option value="cpu">cpu</option>
            </select>
          </div>
        </div>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label class="block text-sm font-medium">FPS (optional)</label>
            <input id="fps" type="number" min="1" class="mt-1 block w-full" />
          </div>
          <div>
            <label class="block text-sm font-medium">Codec</label>
            <select id="codec" class="mt-1 block w-full">
              <option value="h264_videotoolbox">h264_videotoolbox</option>
              <option value="libx264">libx264</option>
            </select>
          </div>
          <div>
            <label class="block text-sm font-medium">Hugging Face token</label>
            <input id="hfToken" type="password" placeholder="hf_..." class="mt-1 block w-full" />
          </div>
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
  const overlayA = document.getElementById('overlayA').files[0]
  const overlayB = document.getElementById('overlayB').files[0]
  const position = document.getElementById('position').value
  const ratio = document.getElementById('ratio').value
  const device = document.getElementById('device').value
  const fps = document.getElementById('fps').value
  const codec = document.getElementById('codec').value
  const hfToken = document.getElementById('hfToken').value

  data.append('video', video)
  if (overlayA) data.append('overlayA', overlayA)
  if (overlayB) data.append('overlayB', overlayB)
  data.append('position', position)
  data.append('overlayWidthRatio', ratio)
  if (fps) data.append('fps', fps)
  data.append('codec', codec)
  data.append('device', device)
  if (hfToken) data.append('hfToken', hfToken)

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


