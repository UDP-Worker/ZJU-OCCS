'use client'
import { useEffect, useState } from 'react'
import { RadioGroup } from '@headlessui/react'
import { Line } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Legend,
} from 'chart.js'
import { CheckCircleIcon } from '@heroicons/react/24/solid'

ChartJS.register(LineElement, PointElement, LinearScale, CategoryScale, Legend)

export default function Home() {
  const [mode, setMode] = useState<'mock' | 'real'>('mock')
  const [voltages, setVoltages] = useState<number[]>([])
  const [curve, setCurve] = useState<{wavelengths:number[]; response:number[]; ideal:number[]}>({wavelengths:[],response:[],ideal:[]})
  const [vRange, setVRange] = useState<[number, number]>([0,2])

  useEffect(() => {
    fetch('/config').then(res => res.json()).then(cfg => {
      setVoltages(Array(cfg.num_channels).fill((cfg.v_range[0]+cfg.v_range[1])/2))
      setVRange(cfg.v_range)
    })
  }, [])

  useEffect(() => {
    if (voltages.length) simulate()
  }, [voltages])

  const simulate = async () => {
    const resp = await fetch('/simulate', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(voltages)
    })
    setCurve(await resp.json())
  }

  const runOptimize = async () => {
    const resp = await fetch('/optimize', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(mode)
    })
    const data = await resp.json()
    setVoltages(data.voltages)
    setCurve(data)
  }

  return (
    <main className="p-4 max-w-3xl mx-auto">
      <h1 className="text-2xl font-bold mb-4 flex items-center">光学滤波器参数寻优 <CheckCircleIcon className="w-6 h-6 text-green-500 ml-2"/></h1>
      <RadioGroup value={mode} onChange={setMode} className="flex space-x-4 mb-4">
        <RadioGroup.Option value="mock" className="cursor-pointer px-3 py-1 rounded bg-gray-200 ui-checked:bg-blue-500 ui-checked:text-white">模拟</RadioGroup.Option>
        <RadioGroup.Option value="real" className="cursor-pointer px-3 py-1 rounded bg-gray-200 ui-checked:bg-blue-500 ui-checked:text-white">实际</RadioGroup.Option>
      </RadioGroup>
      {voltages.map((v, i) => (
        <div className="mb-2" key={i}>
          <label className="mr-2">通道{i+1}</label>
          <input type="range" min={vRange[0]} max={vRange[1]} step="0.01" value={v} className="w-60 align-middle" onChange={e => {
            const arr = [...voltages]
            arr[i] = parseFloat(e.target.value)
            setVoltages(arr)
          }}/>
          <span className="ml-2 w-12 inline-block text-right">{v.toFixed(2)}</span>
        </div>
      ))}
      <button onClick={runOptimize} className="mt-2 px-4 py-1 bg-blue-600 text-white rounded">开始优化</button>
      <div className="mt-6">
        <Line data={{
          labels: curve.wavelengths,
          datasets: [
            {label:'理想波形', data: curve.ideal, borderColor:'blue', fill:false},
            {label:'响应曲线', data: curve.response, borderColor:'red', fill:false}
          ]
        }} options={{responsive:true, interaction:{mode:'index', intersect:false}}}/>
      </div>
    </main>
  )
}
