use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use std::{
    error::Error,
    io,
    time::{Duration, Instant},
};
use tui::{
    backend::{Backend, CrosstermBackend},
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols,
    text::{Span, Spans},
    widgets::canvas::{Canvas, Line, Map, MapResolution, Rectangle},
    widgets::{
        Axis, BarChart, Block, Borders, Cell, Chart, Dataset, Gauge, LineGauge, List, ListItem,
        Paragraph, Row, Sparkline, Table, Tabs, Wrap, GraphType,
    },
    Frame,
    Terminal,
};

extern crate portaudio;
use portaudio as pa;

extern crate ringbuf;

use std::sync::Arc;
use rustfft::{Fft, FftPlanner, num_complex::Complex};

extern crate cubic_spline;
use cubic_spline::{Points, Point, SplineOpts, TryFrom};



const SAMPLE_RATE: f64 = 48_000.0;
const FRAMES: usize = 2048;           
const RINGBUF_SIZE: usize = FRAMES * 8;
const CHANNELS: i32 = 1;
const INTERLEAVED: bool = true;

pub type AudioRingBuffer = ringbuf::HeapRb::<f32>;
type AudioConsumer = ringbuf::Consumer<f32, Arc<AudioRingBuffer>>;
type AudioProducer = ringbuf::Producer<f32, Arc<AudioRingBuffer>>;

struct AudioBackend {
    

    pa: pa::PortAudio,
    stream: Option<pa::Stream<pa::NonBlocking, pa::Input<f32>>>,
    rb_consumer: AudioConsumer,
    buffer: [f32; FRAMES],
    fft: Arc<dyn Fft<f32>>,
    fft_buffer: Vec<Complex<f32>>
}

impl AudioBackend {
    fn new() -> AudioBackend {
        // todo, check result error...
        let pa = pa::PortAudio::new().unwrap();
        let def_input = pa.default_input_device().unwrap();
        let input_info = pa.device_info(def_input).unwrap();
        
        // create input device.__rust_force_expr!
        let latency = input_info.default_low_input_latency;
        let input_params = pa::StreamParameters::<f32>::new(def_input, CHANNELS, INTERLEAVED, latency);
    
        let settings = pa::InputStreamSettings::new(input_params,SAMPLE_RATE, FRAMES as u32);

        // We'll use this channel to send the count_down to the main thread for fun.
        //let (msg_sender, msg_receiver) = ::std::sync::mpsc::channel::<T>();
        
        let rb = ringbuf::HeapRb::<f32>::new(RINGBUF_SIZE);
        let (mut rb_producer, rb_consumer) = rb.split();

        let callback = move |pa::InputStreamCallbackArgs {
            buffer,
            frames,
            flags,
            time,
            ..
            }| 
        {
            assert!(frames == FRAMES as usize);

            if !rb_producer.is_full() {
                rb_producer.push_slice(buffer);
            }

            return pa::Continue;
        };

        let stream = Some(pa.open_non_blocking_stream(settings, callback).unwrap());

        let mut fft_planner = FftPlanner::<f32>::new();
        let fft = fft_planner.plan_fft_forward(FRAMES);

        let audio_backend = AudioBackend {
            pa: pa,
            stream: stream,
            rb_consumer: rb_consumer,
            buffer: [0.0 ; FRAMES],
            fft: fft,
            fft_buffer: vec![Complex{ re: 0.0, im: 0.0 }; FRAMES],
        };
        
        return audio_backend;
    }

    fn start(&mut self) {
        self.stream.as_mut().unwrap().start();
    }

    fn on_tick(&mut self) {       
               
        if !self.rb_consumer.is_empty() {

            let max_frames = self.rb_consumer.pop_slice(&mut self.buffer);

            // setup for fft.
            for i in 0..FRAMES {
                self.fft_buffer[i].re = self.buffer[i];
                self.fft_buffer[i].im = 0.0;
            }

            // windowing todo

            // process fft.
            self.fft.process(&mut self.fft_buffer);

        }

    }

}

struct App {
    audio_backend: AudioBackend,
    data_raw: Vec<(f64, f64)>,
    data_fft_0: Vec<(f64, f64)>,
    data_fft_1: Vec<(f64, f64)>,
    data_fft_2: Vec<(f64, f64)>,
    window: [f64; 2],
}

impl App {
    fn new(audio_backend: AudioBackend) -> App {
        let data_raw = Vec::<(f64, f64)>::new();
        let mut data_fft_0 = Vec::<(f64, f64)>::new();
        let mut data_fft_1 = Vec::<(f64, f64)>::new();
        let mut data_fft_2 = Vec::<(f64, f64)>::new();

        data_fft_0.resize(FRAMES, (0.0, 0.0));
        data_fft_1.resize(FRAMES, (0.0, 0.0));
        data_fft_2.resize(FRAMES, (0.0, 0.0));

        let mut app = App {
            audio_backend: audio_backend,
            data_raw: data_raw,
            data_fft_0: data_fft_0,
            data_fft_1: data_fft_1,
            data_fft_2: data_fft_2,
            window: [0.0, FRAMES as f64],
        };

        app.audio_backend.start();

        return app;
    }

    fn on_tick(&mut self) {
        self.audio_backend.on_tick();

        let mut idx: usize = 0;
        self.data_raw.clear();
        for frame in self.audio_backend.buffer {
            let mut data = (idx as f64, frame as f64);
            
            self.data_raw.push(data);

            idx += 1;
        }

        idx = 0;
        for frame in self.audio_backend.fft_buffer.as_mut_slice() {
            let mag = 10.0 * f32::log10((frame.re * frame.re) + (frame.im * frame.im) + 1e-20);
            let data = (idx as f64, mag as f64);           
            self.data_fft_0[idx] = data;
            idx += 1;
        }

        // cubic spline
        let opts = SplineOpts::new()
        .tension(0.5);

        let mut points = Points::from(&self.data_fft_0[0..(FRAMES/8)]);
        let result = points.calc_spline(&opts).expect("cant construct spline points");

        for i in 0..FRAMES {
            self.data_fft_0[i].0 = result.get_ref()[i].x;
            self.data_fft_0[i].1 = result.get_ref()[i].y;
        }

        // Smooth history for vis.
        for i in 0..FRAMES {
            let data = self.data_fft_0[i];
            self.data_fft_1[i] = (data.0, data.1.max(data.1 * 0.10 + self.data_fft_1[i].1 * 0.90));
            self.data_fft_2[i] = (data.0, data.1.max(data.1 * 0.05 + self.data_fft_2[i].1 * 0.95));
        }
    }


    fn find_strongest_peak(&self, max_x: f64) -> (f64, f64) {
        let mut peak = (0.0, 0.0);
        for i in 0..FRAMES {
            let data = self.data_fft_0[i];
            if data.0 > max_x {
                break;
            }
            if data.1 > peak.1 {
                peak = data;
            }
        }
        return peak;
    }

}

fn main() -> Result<(), Box<dyn Error>> {
    // setup audio backend
    let audio_backend = AudioBackend::new();

    // setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, DisableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // create app and run it
    let tick_rate = Duration::from_micros(16667);
    let app = App::new(audio_backend);
    let res = run_app(&mut terminal, app, tick_rate);

    // restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    if let Err(err) = res {
        println!("{:?}", err)
    }

    Ok(())
}

fn run_app<B: Backend>(
    terminal: &mut Terminal<B>,
    mut app: App,
    tick_rate: Duration,
) -> io::Result<()> {
    let mut last_tick = Instant::now();
    terminal.clear();
    loop {
        terminal.draw(|f| ui(f, &app))?;

        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or_else(|| Duration::from_secs(0));
        if crossterm::event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                if let KeyCode::Char('q') = key.code {
                    return Ok(());
                }
            }
        }
        if last_tick.elapsed() >= tick_rate {
            app.on_tick();
            last_tick = Instant::now();
        }
    }
}

fn ui<B: Backend>(f: &mut Frame<B>, app: &App) {
    let size = f.size();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(
            [
                Constraint::Ratio(9, 10),
                Constraint::Ratio(1, 10),
            ]
            .as_ref(),
        )
        .split(size);

    let fft_bounds = [0.0, (FRAMES / 64) as f64];
    let fft_window = [0.0, (fft_bounds[1] / FRAMES as f64) * SAMPLE_RATE ];

    let x_labels = vec![
        Span::styled(
            format!("{}Hz", fft_window[0]),
            Style::default().add_modifier(Modifier::BOLD),
        ),
        Span::raw(format!("{}Hz", (fft_window[0] * 0.9 + fft_window[1] * 0.1))),
        Span::raw(format!("{}Hz", (fft_window[0] * 0.8 + fft_window[1] * 0.2))),
        Span::raw(format!("{}Hz", (fft_window[0] * 0.7 + fft_window[1] * 0.3))),
        Span::raw(format!("{}Hz", (fft_window[0] * 0.6 + fft_window[1] * 0.4))),
        Span::raw(format!("{}Hz", (fft_window[0] * 0.5 + fft_window[1] * 0.5))),
        Span::raw(format!("{}Hz", (fft_window[0] * 0.4 + fft_window[1] * 0.6))),
        Span::raw(format!("{}Hz", (fft_window[0] * 0.3 + fft_window[1] * 0.7))),
        Span::raw(format!("{}Hz", (fft_window[0] * 0.2 + fft_window[1] * 0.8))),
        Span::raw(format!("{}Hz", (fft_window[0] * 0.1 + fft_window[1] * 0.9))),
        Span::styled(
            format!("{}Hz", fft_window[1]),
            Style::default().add_modifier(Modifier::BOLD),
        ),
    ];

    let datasets = vec![
        Dataset::default()
            .name("FFT 2")
            .marker(symbols::Marker::Braille)
            .style(Style::default().fg(Color::DarkGray))
            .graph_type(GraphType::Line)
            .data(&app.data_fft_2),
        Dataset::default()
            .name("FFT 1")
            .marker(symbols::Marker::Braille)
            .style(Style::default().fg(Color::Gray))
            .graph_type(GraphType::Line)
            .data(&app.data_fft_1),
        Dataset::default()        
            .name("FFT 0")
            .marker(symbols::Marker::Braille)
            .style(Style::default().fg(Color::White))
            .graph_type(GraphType::Line)
            .data(&app.data_fft_0),
    ];

    let chart = Chart::new(datasets)
        .block(
            Block::default()
                .title(Span::styled(
                    "FFT",
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ))
                .borders(Borders::ALL),
        )
        .x_axis(
            Axis::default()
                .title("X Axis")
                .style(Style::default().fg(Color::Gray))
                .labels(x_labels)
                .bounds(fft_bounds),
        )
        .y_axis(
            Axis::default()
                .title("Y Axis")
                .style(Style::default().fg(Color::Gray))
                .labels(vec![
                    Span::styled("0", Style::default().add_modifier(Modifier::BOLD)),
                    Span::raw("0"),
                    Span::styled("50", Style::default().add_modifier(Modifier::BOLD)),
                ])
                .bounds([0.0, 50.0]),
        );
   f.render_widget(chart, chunks[0]);

   let block = Block::default()
        .title(Span::styled(
            "Analysis",
            Style::default()
                .fg(Color::White)
                .bg(Color::Red)
                .add_modifier(Modifier::BOLD),
        ));

    let strongest_peak = app.find_strongest_peak(FRAMES as f64 / 64.0);
    let strongest_pitch = (strongest_peak.0 / FRAMES as f64) * SAMPLE_RATE;
    let fft_window = [0.0, (fft_bounds[1] / FRAMES as f64) * SAMPLE_RATE ];
    
    let text = vec![
            Spans::from(format!("Pitch: {}Hz", strongest_pitch)),
            Spans::from(""),
        ];
    let paragraph = Paragraph::new(text).block(block).wrap(Wrap { trim: true });
    f.render_widget(paragraph, chunks[1]);
}